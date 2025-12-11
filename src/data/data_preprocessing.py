from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

from configs.configs import PathsConfig, run_config


def get_preprocessed_data(spark: SparkSession) -> DataFrame:
    """
    Loads and preprocesses accelerometer data with TAC readings.
    Returns a DataFrame ready for downstream analysis.
    """

    data = spark.read.parquet(PathsConfig.accelerometer_with_tac_parquet_path)
    data = data.sample(fraction=run_config.sample_rate, seed=run_config.random_seed) # optional: right now sample_rate set to 1.0

    return data
