import os
import pickle
import tempfile

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from configs.configs import ModelConfig, run_config

class RandomForestModel:
    def __init__(self):
        self._model = None
        self._cv_scores = None

    @staticmethod 
    def _get_model() -> RandomForestClassifier:
        return RandomForestClassifier(max_depth=ModelConfig.max_depth, n_estimators=ModelConfig.n_estimators, random_state=ModelConfig.random_seed)
    
    @staticmethod
    def _remove_non_training_features(data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop(columns=["pid", "TAC_Reading"])
        return data

    def _get_x_y(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        data = self._remove_non_training_features(data)
        return data.drop("is_intoxicated", axis=1), data["is_intoxicated"].astype(bool)

    def train_model(self, data) -> None:
        x, y = self._get_x_y(data)
        classifier = self._get_model()
        self._cv_scores = cross_validate(
            classifier,
            x,
            y,
            cv=run_config.num_folds,
            return_train_score=True,
            n_jobs=-1, # uses all cores
            scoring=["precision", "recall", "f1", "accuracy"],
        )
        
        self._model = classifier.fit(x, y)

    def get_cv_scores(self) -> dict[str, float]:
        return self._cv_scores

    def load_model_from_mlflow(self, run_id: str) -> None:  # pragma: no cover # TODO why does this work? also put it in the one line of drift calculation
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

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return self._model.predict(features)