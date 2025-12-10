from datetime import date

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

from configs.configs import PathsConfig, run_config, DataConfig


def _timestep_to_seconds(data: DataFrame) -> DataFrame:
    """
    Converts the accelerometer timestamp column ('time') to seconds.
    """
    return data.withColumn(
        "time",
        (F.col("time") / 1000).cast("long")  # Cast to long for integer seconds
    )


def _get_data_windowed(data: DataFrame, time_col: str = "time", window_size_seconds: int = 10) -> DataFrame:
    """
    Adds a windowed_time column based on a timestamp column.

    :param time_col: Name of the timestamp column to use.
    :param window_size_seconds: Size of the time window in seconds.
    :return: DataFrame with 'windowed_time' column
    """
    return data.withColumn(
        "windowed_time",
        F.floor(F.col(time_col).cast("long") / window_size_seconds) * window_size_seconds
    )


def _get_tac_data(spark: SparkSession) -> DataFrame: # TODO rmeove this funciotn, is a oneliner
    """
    Reads the TAC (Transdermal Alcohol Concentration) data from Parquet.
    Keeps the timestamp column as 'timestamp'.
    """
    tac_data = spark.read.parquet(PathsConfig.tac_parquet_path).select(
        "pid", "timestamp", "TAC_Reading"
    )
    return tac_data


def _add_tac_reading(data: DataFrame, tac_data: DataFrame) -> DataFrame:
    """
    Adds the closest previous TAC_Reading to each accelerometer row based on PID.
    Works like the Pandas get_tac_value function: finds the nearest TAC timestamp <= accel time.
    """

    w = Window.partitionBy("pid").orderBy("timestamp")

    joined = data.join(tac_data, on="pid", how="left")

    # Use last() with window to get the latest TAC_Reading <= accel time
    joined = joined.withColumn(
        "TAC_Reading",
        F.last("TAC_Reading", ignorenulls=True).over(
            w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
    )

    return joined


def get_preprocessed_data(spark: SparkSession) -> DataFrame:
    """
    Loads and preprocesses accelerometer data with TAC readings.
    Returns a DataFrame ready for downstream analysis.
    """

    # Load accelerometer data
    data = spark.read.parquet(PathsConfig.accelerometer_with_tac_parquet_path)

    
    # TODO remove these comment

    # data = data.filter(F.col(DataConfig.partition_column).isin(pid_list))


    # data = _timestep_to_seconds(data)
    # data = _get_data_windowed(data, time_col=DataConfig.acceleleration_time_column, window_size_seconds=DataConfig.window_size_seconds)

    # tac_data = _get_tac_data(spark)

    # data = data.sample(run_config.sample_rate, seed=run_config.random_seed)

    # data = _add_tac_reading(data, tac_data)


    return data
