import os
import pickle
import tempfile
import typing

from pyspark.sql import DataFrame
import mlflow
from pyspark.ml.feature import StringIndexerModel, OneHotEncoderModel

from configs.configs import run_config
from pendulum import datetime

from pyspark.sql import functions as F

class _FeatureExtractorData: # TODO check this code and know what it does
    """
    Class for holding the data for FeatureExtractor
    """
    def __init__(self): # TODO
        self._mean_energy: typing.Optional[dict[str, float]] = {}
        self._std_energy: typing.Optional[dict[str, float]] = {}
        self._mean_magnitude: typing.Optional[dict[str, float]] = {}
        self._std_magnitude: typing.Optional[dict[str, float]] = {}
    
    def is_set(self) -> bool: 
        return True if None not in self.__getstate__().values() and {} not in self.__getstate__().values() else False

    def save(self, directory_path: str):
        with open(os.path.join(directory_path, "mean_energy.pkl"), "wb") as f:
            pickle.dump(self._mean_energy, f)
        with open(os.path.join(directory_path, "std_energy.pkl"), "wb") as f:
            pickle.dump(self._std_energy, f)
        with open(os.path.join(directory_path, "mean_magnitude.pkl"), "wb") as f:
            pickle.dump(self._mean_magnitude, f)
        with open(os.path.join(directory_path, "std_magnitude.pkl"), "wb") as f:
            pickle.dump(self._std_magnitude, f)

    def load_from_mlflow(self, run_id: str):  # pragma: no cover
        """
        Loads the state from mlflow from the artifacts of the Run of given RunID

        :raises RuntimeError: if experiment does not exist.
        :raises Exception: if given run_id is not in the expected mlflow experiment.
        """
        # WARNING: Beware that I realized in WSL, the temp directory is not cleaned as it should. This may result in
        # unexpected issues with loading and saving.
        with tempfile.TemporaryDirectory() as dir_name:
            mlflow_experiment = mlflow.get_experiment_by_name(run_config.experiment_name)
            if mlflow_experiment is None:
                raise RuntimeError(f"Experiment {run_config.experiment_name} does not exist in MLFlow.")
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=run_config.run_name, dst_path=dir_name)

            # mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=dir_name) # TODO kijk hier naar 
            artifacts_path = os.path.join(dir_name, run_config.run_name)


            with open(os.path.join(artifacts_path, "mean_energy.pkl"), "rb") as f:
                self._mean_energy = pickle.load(f)
            with open(os.path.join(artifacts_path, "std_energy.pkl"), "rb") as f:
                self._std_energy = pickle.load(f) 
            with open(os.path.join(artifacts_path, "mean_magnitude.pkl"), "rb") as f:
                self._mean_magnitude = pickle.load(f)
            with open(os.path.join(artifacts_path, "std_magnitude.pkl"), "rb") as f:
                self._std_magnitude = pickle.load(f)

    def save_to_mlflow(self, run_id: str):  
        """
        Saves the state as an artifact to mlflow, inside the given Run with RunID

        :raises RuntimeError: if experiment does not exist.
        :raises Exception: if given run_id is not in the expected mlflow experiment.
        """
        with tempfile.TemporaryDirectory() as dir_name:
            mlflow_experiment = mlflow.get_experiment_by_name(run_config.experiment_name)
            if mlflow_experiment is None:
                raise RuntimeError(f"Experiment {run_config.experiment_name} does not exist in MLFlow.")
            self.save(dir_name)
            mlflow.log_artifacts(dir_name, run_config.run_name, run_id)

        
class FeatureExtractor:
    def __init__(self):
        self._state = _FeatureExtractorData()
    
    def save_to_mlflow(self, run_id: str):
        self._state.save_to_mlflow(run_id)

    def load_from_mlflow(self, run_id: str):
        self._state.load_from_mlflow(run_id)

    def get_features(self, data: DataFrame) -> DataFrame:
        is_inference_time = self._state.is_set()
        return self.get_inference_features(data) if is_inference_time else self.get_training_features(data)
    
    def get_training_features(self, data: DataFrame) -> DataFrame: # TODO sla de features op in mlflow want daarna werkt die is set ding van hem wel 

        data = self._get_energy(data)
        data = self._get_magnitude(data)
        data = self._get_is_intoxicated(data, threshold=run_config.intoxication_threshold) # TODO remove?

        # TODO meer met clean coding dit doen!
        self._set_mean_energy(
            data.select(F.mean(F.col("energy"))).collect()[0][0], data.select(F.std(F.col("energy"))).collect()[0][0]
        )
        self._set_mean_magnitude(
            data.select(F.mean(F.col("magnitude"))).collect()[0][0], data.select(F.std(F.col("magnitude"))).collect()[0][0]
        )
        self._set_std_energy(
            data.select(F.std(F.col("energy"))).collect()[0][0], data.select(F.std(F.col("energy"))).collect()[0][0]
        )
        self._set_std_magnitude(
            data.select(F.std(F.col("magnitude"))).collect()[0][0], data.select(F.std(F.col("magnitude"))).collect()[0][0]
        )

        return data
    
    def get_inference_features(self, data: DataFrame) -> DataFrame:
        data = self._get_energy(data)
        data = self._get_magnitude(data)
        #TODO should i add the means and stuff here as well?
        return data
    
    def _get_is_intoxicated(self, data: DataFrame, threshold: float) -> DataFrame:
        return data.withColumn("is_intoxicated", F.col("TAC_Reading") >= threshold)
    

    # TODO maybe an is night feature?
    def _get_time_of_day(self, timestamp: int) -> str: # NOTE: moet dan met onehotencoding en die stringindexer van ...
        """
        Extract time of day feature from timestamp

        returns: str: 'morning', 'afternoon', 'evening', 'night'
        """
        hour = datetime.utcfromtimestamp(timestamp).hour

        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
        
    def _get_energy(self, data: DataFrame) -> DataFrame: # TODO fix comment maybe to chatty
        """
        Compute energy feature from accelerometer data

        returns: DataFrame: with energy feature added
        """
        return (
        data.withColumn(
            "energy",
            F.col("x") * F.col("x") +
            F.col("y") * F.col("y") +
            F.col("z") * F.col("z")
        )
    )

    def _get_mean_energy(self, data: DataFrame) -> DataFrame: # TODO fix die window id column
        """
        Computes: mean energy per window.

        Requires `energy` and `window_id` columns.

        returns: DataFrame with mean_energy added
        """
        return (
            data.groupBy("window_id")
                .agg(F.mean("energy").alias("mean_energy"))
        )
    
    def _get_magnitude(self, data: DataFrame) -> DataFrame: # TODO fix comments
        """
        Compute magnitude = sqrt(x^2 + y^2 + z^2)

        returns: DataFrame with 'magnitude' feature added
        """
        return data.withColumn("magnitude", F.sqrt(F.col("energy")))
    
    def _set_mean_energy(self, mean: float, std: float) -> None:
        self._state._mean_energy = {"mean": mean, "std": std}
    
    def _set_std_energy(self, mean: float, std: float) -> None:
        self._state._std_energy = {"mean": mean, "std": std}

    def _set_mean_magnitude(self, mean: float, std: float) -> None:
        self._state._mean_magnitude = {"mean": mean, "std": std}

    def _set_std_magnitude(self, mean: float, std: float) -> None:
        self._state._std_magnitude = {"mean": mean, "std": std}


    
    

