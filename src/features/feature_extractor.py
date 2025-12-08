import os
import pickle
import tempfile
import typing

from pyspark.sql import DataFrame, SparkSession
import mlflow
from pyspark.ml.feature import StringIndexerModel, OneHotEncoderModel

from configs.configs import run_config, PathsConfig
from pendulum import datetime
import numpy as np

from pyspark.sql import functions as F

class _FeatureExtractorData: # TODO check this code and know what it does
    """
    Class for holding the data for FeatureExtractor
    """
    def __init__(self): # TODO
        self._mean_per_feature: typing.Optional[dict[str, float]] = {}
        self._std_per_feature: typing.Optional[dict[str, float]] = {}
        self._window_size_seconds: typing.Optional[float] = None


    def save(self, directory_path: str):
        with open(os.path.join(directory_path, "mean_per_feature.pkl"), "wb") as f:
            pickle.dump(self._mean_per_feature, f)
        with open(os.path.join(directory_path, "std_per_feature.pkl"), "wb") as f:
            pickle.dump(self._std_per_feature, f)
        with open(os.path.join(directory_path, "window_size_seconds.pkl"), "wb") as f:
            pickle.dump(self._window_size_seconds, f)

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


            with open(os.path.join(dir_name, "mean_per_feature.pkl"), "rb") as f:
                self._mean_per_feature = pickle.load(f)
            with open(os.path.join(dir_name, "std_per_feature.pkl"), "rb") as f:
                self._std_per_feature = pickle.load(f)
            with open(os.path.join(dir_name, "window_size_seconds.pkl"), "rb") as f:
                self._window_size_seconds = pickle.load(f)  


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
        self._data = _FeatureExtractorData()
    
    def save_to_mlflow(self, run_id: str):
        self._data.save_to_mlflow(run_id)

    def load_from_mlflow(self, run_id: str):
        self._data.load_from_mlflow(run_id)

    def get_features(self, data: DataFrame) -> DataFrame:
        data = self._get_energy(data)
        # data = self._get_mean_energy(data) 
        data = self._get_magnitude(data)

        return data  # TODO maybe implement inference and training seperately?
    
    def _get_last_tac_given_time(self, data: DataFrame, pid: str, timestamp: int) -> float: # TODO maybe in preprocessing?
        """
        Get the last TAC reading before a given timestamp for a patient

        returns: float: the last TAC reading
        """

        pid_df = data[pid]
        
        closest_idx = np.argmax(pid_df['timestamp'] > timestamp)

        if closest_idx != 0: # adjust index iff not the first element
            closest_idx -= 1

        return pid_df.at[closest_idx, 'TAC_Reading'] # retrieves the TAC reading at the closest index
    

    def _is_intoxicated(self, tac_reading: float, threshold: float) -> bool: # TODO apply this as extra feature maybe use as label?
        return tac_reading >= threshold
    

    # TODO maybe an is night feature?
    def _time_of_day_feature(self, timestamp: int) -> str:
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

    
    

