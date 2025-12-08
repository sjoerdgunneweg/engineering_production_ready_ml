from datetime import date

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

from configs.configs import PathsConfig, run_config

def _get_data_windowed(data: DataFrame, window_size_seconds: int = 10) -> DataFrame:
    """
    Reads the preprocessed Parquet data and applies a time windowing function.

    :param window_size_seconds: Size of the time window in seconds.
    :return: Windowed DataFrame
    """
    windowed_data = data.withColumn(
        "windowed_time",
        F.floor(F.col("time").cast("long") / window_size_seconds) * window_size_seconds
    )

    return windowed_data

# def _get_data_from_parquet(parquet_path: str) -> DataFrame: # TODO remove maybe
#     """
#     Reads the accelerometer data for all participants.

#     :return: DataFrame with accelerometer data
#     """
#     spark = SparkSession.builder.master("local[2]").getOrCreate()

#     return spark.read.parquet(parquet_path)

def _get_tac_data(spark: SparkSession) -> DataFrame: # TODO fix comment
    """
    Reads the Transdermal Alcohol Concentration (TAC) data from Parquet files.

    :return: DataFrame with TAC data
    """

    tac_data = spark.read.parquet(PathsConfig.tac_parquet_path).withColumnRenamed("timestamp", "tac_time_sec") # rename to avoid confusion

    tac_data = tac_data.select("pid", "tac_time_sec", "TAC_Reading")
        
    return tac_data.sort(F.col("pid"), F.col("tac_time_sec")) # TODO explain why sorted
    

# def _add_tac_reading(data: DataFrame, tac_data: DataFrame) -> DataFrame: # TODO fix comment
#     """
#     Adds the Transdermal Alcohol Concentration (TAC) reading to the accelerometer data 
#     using Last Observation Carried Forward (LOCF) logic.
#     For each data window, it finds the latest TAC reading taken AT or BEFORE the windowed time.
#     """
    
#     # 1. Join Accelerometer data (A) with TAC data (T) 
#     # Condition: Match PID AND A's windowed time is GREATER THAN or EQUAL TO T's TAC time.
#     joined_data = data.alias("A").join(
#         tac_data.alias("T"),
#         (F.col("A.pid") == F.col("T.pid")) & (F.col("A.windowed_time") >= F.col("T.tac_time_sec")),
#         "left_outer"
#     )

#     # 2. Define a Window to find the most recent (largest tac_time_sec) TAC reading for each data window
#     # Partition by PID and the accelerometer data window. Order by TAC time descending to put the latest reading first.
#     window_spec = Window.partitionBy("A.pid", "A.windowed_time").orderBy(F.col("T.tac_time_sec").desc())

#     # 3. Apply the window function to rank the matching TAC readings
#     ranked_data = joined_data.withColumn("rank", F.row_number().over(window_spec))

#     # 4. Filter for the latest reading (rank 1) and select the final columns
#     tac_added_data = ranked_data.filter(F.col("rank") == 1).select(
#         "A.*",
#         F.col("T.TAC_Reading").alias("TAC_reading")
#     )
    
#     return tac_added_data

def _add_tac_reading(data: DataFrame, tac_data: DataFrame) -> DataFrame:
    """
    Efficiently adds TAC readings to accelerometer data using LOCF (Last Observation Carried Forward)
    without performing an expensive non-equi join.

    Steps:
    1. Window TAC readings by PID and time.
    2. Forward-fill TAC values with last() over time.
    3. Assign TAC readings to the same time windows as accelerometer data.
    4. Join on (pid, windowed_time).
    """

    # 1. Ensure both dataframes are partitioned by PID
    data = data.repartition("pid")
    tac_data = tac_data.repartition("pid")

    # 2. Create a window that orders TAC readings within each participant
    tac_window = (
        Window.partitionBy("pid")
              .orderBy(F.col("tac_time_sec"))
              .rowsBetween(Window.unboundedPreceding, 0)   # LOCF window
    )

    # 3. Forward-fill TAC readings
    tac_filled = tac_data.withColumn(
        "TAC_reading_filled",
        F.last("TAC_Reading", ignorenulls=True).over(tac_window)
    )

    # 4. Align TAC times to the same 10-second windows as accelerometer data
    tac_windowed = tac_filled.withColumn(
        "windowed_time",
        F.floor(F.col("tac_time_sec") / 10) * 10
    ).select(
        "pid",
        "windowed_time",
        "TAC_reading_filled"
    )

    # 5. Join on (pid, windowed_time)
    result = data.join(
        tac_windowed,
        on=["pid", "windowed_time"],
        how="left"
    ).withColumnRenamed("TAC_reading_filled", "TAC_reading")

    return result


def get_preprocessed_data(spark: SparkSession) -> DataFrame: # TODO fix this comment 
    """
    Preprocessing step.

    Returns: DataFrame with preprocessed data including TAC readings added.
    """
    data = spark.read.parquet(PathsConfig.accelerometer_parquet_path)
    data = _get_data_windowed(data, window_size_seconds=10)

    tac_data = _get_tac_data(spark)
    data = _add_tac_reading(data, tac_data)

    data = data.sample(run_config.sample_rate, seed=run_config.random_seed)
    return data