from functools import cache
import logging

import mlflow
from http import HTTPStatus
from pyspark.sql import SparkSession
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter
from flask import Flask, request, jsonify


from configs.configs import run_config, ModelConfig
from src.features.feature_extractor import FeatureExtractor
from src.utils.mlflow_utils import get_latest_run_id
from src.random_forest import RandomForestModel

logger = logging.getLogger(__name__)

@cache
def get_model() -> RandomForestModel:
    mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
    run_id = get_latest_run_id(run_config.run_name)
    model = RandomForestModel()
    model.load_model_from_mlflow(run_id)
    return model


@cache
def get_feature_extractor_loaded() -> FeatureExtractor:
    mlflow.set_tracking_uri(run_config.mlflow_tracking_uri)
    run_id = get_latest_run_id(run_config.run_name)
    feature_extractor = FeatureExtractor()
    feature_extractor.load_from_mlflow(run_id)
    return feature_extractor


logger.info("Loading cache and starting spark session.")
get_model()
get_feature_extractor_loaded()
SparkSession.builder.master(run_config.spark_master_url).getOrCreate()

logger.info("Starting flask app.")
app = Flask(__name__)
metrics = PrometheusMetrics(app, defaults_prefix=run_config.app_name)
metrics.info(
    f"{run_config.app_name}_model_version",
    "Model version information",
    experiment_name=run_config.experiment_name,
    run_name=run_config.run_name,
    model_name=ModelConfig.model_name,
)

pred_counter = Counter(f"{run_config.app_name}_predictions", "Count of predictions by class", labelnames=["pred"]) # TODO what does this do?
data_quality_counter = Counter(
    f"{run_config.app_name}_data_quality", "Count of data quality issues", labelnames=["quality_rule"]
)


@app.route("/health", methods=["GET"])
@metrics.do_not_track()
def health():
    logging.debug("Health endpoint pinged.")
    return "OK", HTTPStatus.OK


@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Received prediction request at /predict.")
    model = get_model()
    print(model)
    print("Model loaded successfully.")
    feature_extractor = get_feature_extractor_loaded()
    spark = SparkSession.builder.master(run_config.spark_master_url).getOrCreate()

    request_data = spark.createDataFrame(
        [
            {
                # Use the correct columns for the alcoholerometer project
                "pid": request.json["pid"],
                "x": request.json["x"],
                "y": request.json["y"],
                "z": request.json["z"],
            },
        ]
    )
    logging.info("Calculating features for the request.")
    features = feature_extractor.get_features(request_data)
    features = features.toPandas()

    features = features.drop(columns=["pid", "is_intoxicated"], errors='ignore') # TODO if this line sovles the issue, make a seperet get_features for inference and training in feature extractor

    logging.info("Features are ready. Calculating prediction.")
    prediction = model.predict(features)[0].item()
    pred_counter.labels(pred="true" if prediction else "false").inc()
    logging.info(f"Prediction complete. Prediction: {prediction}")
    return jsonify(prediction)


@app.route("/reload", methods=["POST"])
def reload():
    # clear the caches to before reloading to make sure new model and feature extractor are actually new
    get_model.cache_clear()
    get_feature_extractor_loaded.cache_clear()

    get_model()
    get_feature_extractor_loaded()    
    return "Reloaded Model Successfully", HTTPStatus.OK