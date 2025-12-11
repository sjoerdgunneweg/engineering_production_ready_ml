import argparse
import logging
import mlflow
import json

from data.data_preprocessing import get_preprocessed_data
from mlflow.entities import RunStatus


from pyspark.sql import SparkSession
from features.feature_extractor import FeatureExtractor
from random_forest import RandomForestModel

from configs.configs import PathsConfig, run_config, ModelConfig
from utils.mlflow_utils import create_mlflow_experiment_if_not_exist, create_mlflow_run_if_not_exists, get_latest_run_id, save_artifacts_to_mlflow

def main(args: argparse.Namespace):

    spark = SparkSession.builder.master("local[*]").getOrCreate()

    if args.preprocess:
        data = get_preprocessed_data(spark) 
        data.write.parquet(PathsConfig.preprocessing_data_path, mode="overwrite")

    if args.feat_eng:
        data = spark.read.parquet(PathsConfig.preprocessing_data_path)
        
        feature_extractor = FeatureExtractor() 
        data = feature_extractor.get_features(data)


        mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
        create_mlflow_experiment_if_not_exist()
        create_mlflow_run_if_not_exists(run_config.run_name)
        feature_extractor.save_to_mlflow(get_latest_run_id(run_config.run_name))
        data.write.parquet(PathsConfig.features_data_path, mode="overwrite")

    if args.training:

        data = spark.read.parquet(PathsConfig.features_data_path).toPandas()

        random_forest_model = RandomForestModel() 
        random_forest_model.train_model(data)
        logging.info(f"Cross-validation scores: {random_forest_model.get_cv_scores()}")

        mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
        create_mlflow_experiment_if_not_exist()
        create_mlflow_run_if_not_exists(run_config.run_name)
        save_artifacts_to_mlflow(
            {ModelConfig.model_name: random_forest_model, "cv_scores": random_forest_model.get_cv_scores()},
            get_latest_run_id(run_config.run_name),
        )
        mlflow.end_run(RunStatus.to_string(RunStatus.FINISHED))

        telemetry_data = { # TODO update to somehting else?
            "is_intoxicated": {
                False: sum(data["is_intoxicated"] == False),
                True: sum(data["is_intoxicated"] == True),
            }
        }
        print(telemetry_data)
        with open(PathsConfig.telemetry_training_data_path, "w") as file:
            json.dump(telemetry_data, file)
            print(f"Written telemetry training data to: {PathsConfig.telemetry_training_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing')
    parser.add_argument('--feat_eng', action='store_true', help='Run feature engineering')
    parser.add_argument('--training', action='store_true', help='Run model training')
    parser.add_argument('--reload', action='store_true', help='Reload the model in the API')
    args = parser.parse_args()
    main(args)