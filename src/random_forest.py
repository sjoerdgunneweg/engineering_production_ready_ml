import os
import pickle

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from configs.configs import ModelConfig, run_config, PathsConfig

class RandomForestModel:
    def __init__(self):
        self._model = None

    @staticmethod # TODO wat doet dit?
    def _get_model() -> RandomForestClassifier:
        return RandomForestClassifier(max_depth=ModelConfig.max_depth, n_estimators=ModelConfig.n_estimators, random_state=ModelConfig.random_seed)
    
    @staticmethod # TODO wat doet dit?
    def _get_x_y(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        # return data.drop("TAC_Reading", axis=1), data["TAC_Reading"] # TODO maybe in configs
        # return data.drop("TAC_Reading"), data["TAC_Reading"] # TODO maybe in configs   
        # 

        data.drop("readable_time", axis=1, inplace=True) # TODO remove this or find better fix
        data.drop("measurement_date", axis=1, inplace=True) # TODO remove this or find better fix
        return data.drop("pid", axis=1), data["pid"]


    def train_model(self, data):
        x, y = self._get_x_y(data)
        classifier = self._get_model()
        
        self._model = classifier.fit(x, y)

    def load_model_from_mlflow(self, run_id: str):  # pragma: no cover
        mlflow_experiment = mlflow.get_experiment_by_name(run_config.experiment_name)
        if mlflow_experiment is None:
            raise RuntimeError(f"Experiment {run_config.experiment_name} does not exist in MLFlow.")

        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                dst_path=temp_dir,
                tracking_uri=run_config.mlflow_tracking_uri,
            )
            artifacts_path = temp_dir + "/" + run_config.run_name
            model_file_name = ModelConfig.model_name + ".pkl"
            if model_file_name not in os.listdir(artifacts_path):
                raise RuntimeError(f"Model {ModelConfig.model_name} is not among MLFLow artifacts.")
            with open(artifacts_path + "/" + model_file_name, "rb") as file:
                self._model = pickle.load(file)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame: # TODO moet dit soort shit niet pysspark?
        return self._model.predict(features)