import os
import pandas as pd
import numpy as np
import re
from pyspark.sql import SparkSession
from typing import Dict, List
from pyspark.sql.types import StructType, StructField, StringType, LongType, FloatType

from configs.configs import DataConfig, PathsConfig
from pyspark.sql import DataFrame

FINAL_SCHEMA = StructType([
    StructField(DataConfig.partition_column, StringType(), True),
    StructField(DataConfig.acceleleration_time_column, LongType(), True), # Window start time in Unix seconds
    StructField(DataConfig.tac_reading_column, FloatType(), True),
    StructField("x", FloatType(), True),
    StructField("y", FloatType(), True),
    StructField("z", FloatType(), True),
    # Include all other features calculated in create_sample_feature_frames here.
])

FINAL_COLUMN_ORDER: List[str] = [field.name for field in FINAL_SCHEMA.fields]


def _consolidate_full_tac_timeseries(input_path: str) -> pd.DataFrame: # TODO rename function
    """
    Reads all raw TAC CSVs, extracts PID, and consolidates the full, 
    unaggregated time series into one Pandas DataFrame.
    
    Returns a single, time-sorted Pandas DataFrame of TAC readings.
    """
    if not os.path.isdir(input_path):
        print(f"TAC data directory not found at: {input_path}") # TODO make logging
        return pd.DataFrame() # return empty DataFrame if path not found
    
    # use scandir, since more efficient than listdir
    tac_files = [os.path.join(input_path, entry.name) for entry in os.scandir(input_path) if entry.is_file() and entry.name.endswith('.csv')]
    
    all_tac_data = []
    pid_regex = r'([A-Z]{2}\d{4})' # regex finding pid pattern like 'AB1234'

    for file_path in tac_files:
        file_name = os.path.basename(file_path)
        match = re.search(pid_regex, file_name) # extract PID from filename
        
        if not match: 
            continue
        
        pid = match.group(1) # Extracted PID
        
        try:
            tac_df = pd.read_csv(file_path, usecols=[DataConfig.tac_time_column, DataConfig.tac_reading_column])
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        tac_df[DataConfig.partition_column] = pid
        # convert timestamp to datetime objects (Verified to be 's')
        tac_df[DataConfig.tac_time_column] = pd.to_datetime(tac_df[DataConfig.tac_time_column], unit='s') 
        
        all_tac_data.append(tac_df)

    if not all_tac_data:
        print("No TAC CSV files found or processed.")
        return pd.DataFrame()
        
    tac_combined_df = pd.concat(all_tac_data, ignore_index=True)
    return tac_combined_df.sort_values([DataConfig.partition_column, DataConfig.tac_time_column])


def add_tac_label_to_windowed_data(feature_frames_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Adds the closest previous TAC_Reading label to each time window 
    """
    
    full_tac_timeseries_pd = _consolidate_full_tac_timeseries(PathsConfig.clean_tac_path)
    
    if full_tac_timeseries_pd.empty:
        print("TAC time series could not be loaded.") # TODO make logging
        return feature_frames_dict

    # group TAC data by PID for fast extraction inside the loop
    tac_pd_by_pid = {
        pid: df
        for pid, df in full_tac_timeseries_pd.groupby(DataConfig.partition_column)
    }

    labeled_frames = {}

    for pid, features_pd in feature_frames_dict.items():
        tac_pd_single_pid = tac_pd_by_pid.get(pid, pd.DataFrame())

        if tac_pd_single_pid.empty:
            print(f"Skipping PID {pid}: No corresponding TAC data found.") #TODO make logging
            labeled_frames[pid] = features_pd 
            continue
            
        # must reset the index to make the time column available for the merge.
        features_pd_reset = features_pd.reset_index()

        # The index name from the create_sample_feature_frames is 'window_start_time'
        time_column_name = features_pd.index.names[0] 
        features_pd_reset['window_time_dt'] = pd.to_datetime(
            features_pd_reset[time_column_name], unit='s'
        )
        
        # NOTE: features_pd_reset MUST be sorted by the datetime column for merge_asof
        features_pd_sorted = features_pd_reset.sort_values('window_time_dt')

        # --- 3. The Time-Series Look-Back Join (Labeling) --- # TODO better comment + explanation
        labeled_pd = pd.merge_asof(
            left=features_pd_sorted, 
            right=tac_pd_single_pid,
            left_on='window_time_dt', # Use the new datetime column
            right_on=DataConfig.tac_time_column,
            by=DataConfig.partition_column, 
            direction='backward'           
        )
        
        
        # Drop temporary columns used for the merge (datetime column and TAC timestamp column)
        labeled_pd = labeled_pd.drop(
            columns=[DataConfig.tac_time_column, 'window_time_dt'], errors='ignore'
        )
        
        # Restore the index using the original time column name.
        labeled_pd = labeled_pd.set_index(time_column_name)

        # ensure the TAC_Reading is float 
        labeled_pd[DataConfig.tac_reading_column] = labeled_pd[DataConfig.tac_reading_column].astype(float)
        
        labeled_frames[pid] = labeled_pd

    return labeled_frames


def create_sample_feature_frames(accel_path: str) -> Dict[str, pd.DataFrame]:
    """
    Simulates the feature engineering step: loads the raw data and groups it by 
    a simplified 10-second window, calculating the mean of x, y, z.
    The resulting DataFrame index is the window start time (in seconds).
    """    
    acceleration_df = pd.read_csv(accel_path)
    
    # convert time from ms to seconds
    acceleration_df['time_s'] = acceleration_df[DataConfig.acceleleration_time_column] / 1000
    
    # create window of 10 seconds. 
    acceleration_df['window_key'] = (acceleration_df['time_s'] // DataConfig.window_size_seconds) * DataConfig.window_size_seconds
    
    feature_frames_dict = {}
    for pid, group_df in acceleration_df.groupby(DataConfig.partition_column):
        
        # calculate mean of x, y, z per window 
        feature_frame = group_df.groupby('window_key')[['x', 'y', 'z']].mean() 
        
        # Add the PID back as a column for the merge_asof 'by' parameter
        feature_frame[DataConfig.partition_column] = pid
        
        # Set the index name, which will be used as the time column in the merge logic
        feature_frame.index.name = 'window_start_time' 
        
        feature_frames_dict[pid] = feature_frame
        
    return feature_frames_dict

def get_preprocessed_data_v2(spark: SparkSession) -> DataFrame:
    """
    Loads and preprocesses accelerometer data with TAC readings.
    Returns a DataFrame ready for downstream analysis.
    """

    spark = (
        SparkSession.builder
        .master("local[*]")
        .config("spark.driver.memory", "8g") # extra memory for this expensive operation
        .getOrCreate()
    )
    
    features_dictionary = create_sample_feature_frames(PathsConfig.accelerometer_data_path)
    features_with_tac_dictionary = add_tac_label_to_windowed_data(features_dictionary)

    final_combined_pd = pd.concat(features_with_tac_dictionary.values())
    
    # convert the index  back
    final_combined_pd = final_combined_pd.reset_index().rename(columns={'window_start_time': DataConfig.acceleleration_time_column})
    final_combined_pd[DataConfig.acceleleration_time_column] = final_combined_pd[DataConfig.acceleleration_time_column].astype(np.int64)

    final_spark_df = spark.createDataFrame(final_combined_pd)

    final_spark_df.write.parquet(
        PathsConfig.accelerometer_with_tac_parquet_path,
        mode="overwrite",
        partitionBy=DataConfig.partition_column
    )

    print(f"Written labeled feature data to: {PathsConfig.accelerometer_with_tac_parquet_path}")

    # Load accelerometer data
    data = spark.read.parquet(PathsConfig.accelerometer_with_tac_parquet_path)
    spark.stop() 


    return data
