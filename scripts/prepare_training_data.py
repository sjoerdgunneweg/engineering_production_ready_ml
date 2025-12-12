import os
import logging
import pandas as pd
import numpy as np
import re
from pyspark.sql import SparkSession
from typing import Dict

from configs.configs import DataConfig, PathsConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _combine_raw_tac_files(input_path: str) -> pd.DataFrame:
    """
    Reads all raw TAC CSVs, extracts PID, and combines them into one DataFrame.
    
    Returns: time-sorted Pandas DataFrame of TAC readings.
    """
    if not os.path.isdir(input_path):
        logger.warning(f"TAC data directory not found at: {input_path}") 
        return pd.DataFrame() # return empty DataFrame if path not found
    
    # use scandir, since more efficient than listdir
    tac_files = [os.path.join(input_path, entry.name) for entry in os.scandir(input_path) if entry.is_file() and entry.name.endswith('.csv')]

    all_tac_data = []

    for file_path in tac_files:
        file_name = os.path.basename(file_path) # only the PID + _clean_TAC.csv part
        match = re.search(DataConfig.pid_regex_pattern, file_name) # extract PID from filename
        
        if not match: # skip files without valid PID
            continue
        
        pid = match.group(1) # Extracted PID
        
        try:
            tac_df = pd.read_csv(file_path, usecols=[DataConfig.tac_time_column, DataConfig.tac_reading_column])
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            continue

        tac_df[DataConfig.partition_column] = pid

        # convert timestamp to datetime objects for easier merging later
        tac_df[DataConfig.tac_time_column] = pd.to_datetime(tac_df[DataConfig.tac_time_column], unit='s') 
        
        all_tac_data.append(tac_df)

    if not all_tac_data:
        logger.warning("No TAC CSV files found or processed.")
        return pd.DataFrame()
        
    tac_combined_df = pd.concat(all_tac_data, ignore_index=True)
    return tac_combined_df.sort_values([DataConfig.partition_column, DataConfig.tac_time_column])


def _add_tac_reading_to_windowed_data(feature_frames_dict: Dict[str, pd.DataFrame], full_tac_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Adds the closest previous TAC_Reading to each time window 

    returns: Dict[str, pd.DataFrame]: dictionary of labeled feature DataFrames per PID # TODO fix comment
    """
    
    if full_tac_df.empty:
        logger.warning("TAC data could not be loaded.")
        return feature_frames_dict

    # group TAC data by PID for fast extraction inside the loop
    tac_pd_by_pid = {pid: df for pid, df in full_tac_df.groupby(DataConfig.partition_column)}

    features_and_tac_by_pid= {}

    for pid, features_pd in feature_frames_dict.items():
        tac_pd_single_pid = tac_pd_by_pid.get(pid)

        if tac_pd_single_pid is None:
            logger.warning(f"Skipping PID {pid}: No corresponding TAC data found.")
            features_and_tac_by_pid[pid] = features_pd 
            continue
            
        # must reset the index to make the time column available for the merge.
        features_pd_reset = features_pd.reset_index()

        features_pd_reset[DataConfig.datetime_column] = pd.to_datetime(features_pd_reset[DataConfig.window_start_index_name], unit='s')
        
        # features_pd_reset must be sorted by the datetime column for merge_asof
        features_pd_sorted = features_pd_reset.sort_values(DataConfig.datetime_column)

        # merge_asof mathes on nearest previous timestamp instead of exact match
        labeled_pd = pd.merge_asof(
            left=features_pd_sorted, 
            right=tac_pd_single_pid,
            left_on=DataConfig.datetime_column, # the time column in the features
            right_on=DataConfig.tac_time_column, # the time column in the TAC data
            by=DataConfig.partition_column, # ensure only matches with same pid
            direction='backward' # only previous matches       
        )
        
        # drop the temporary columns used for the merge
        labeled_pd = labeled_pd.drop(
            columns=[DataConfig.tac_time_column, DataConfig.datetime_column], errors='ignore'
        )
        
        # restore the index using the original time column name.
        labeled_pd = labeled_pd.set_index(DataConfig.window_start_index_name)
        
        features_and_tac_by_pid[pid] = labeled_pd

    return features_and_tac_by_pid


def _calculate_window_features(acceleration_data_path: str) -> Dict[str, pd.DataFrame]:
    """
    Loads the raw acceleration data and groups it by a window of 10 seconds.
    For each window it calculates the mean of x, y, z

    returns: Dict[str, pd.DataFrame]: dictionary of feature DataFrames per PID.
    """    
    acceleration_df = pd.read_csv(acceleration_data_path)
    
    # convert time from ms to seconds
    acceleration_df[DataConfig.time_in_seconds_column] = acceleration_df[DataConfig.acceleleration_time_column] / 1000
    
    # create window of 10 seconds. 
    acceleration_df[DataConfig.window_key_column] = (acceleration_df[DataConfig.time_in_seconds_column] // DataConfig.window_size_seconds) * DataConfig.window_size_seconds
    
    features_by_pid = {}
    for pid, group_df in acceleration_df.groupby(DataConfig.partition_column):
        
        # calculate mean of x, y, z per window 
        pid_features_df = group_df.groupby(DataConfig.window_key_column)[['x', 'y', 'z']].mean() 
        
        # add the PID back as a column for the merge_asof 'by' parameter
        pid_features_df[DataConfig.partition_column] = pid
        
        # set the index name, will be used as the time column in the merge logic    
        pid_features_df.index.name = DataConfig.window_start_index_name
        
        features_by_pid[pid] = pid_features_df
        
    return features_by_pid


def main():
    spark = (
        SparkSession.builder
        .master("local[*]")
        .config("spark.driver.memory", "8g") # extra memory for this expensive operation
        .getOrCreate()
    )

    full_tac_df = _combine_raw_tac_files(PathsConfig.clean_tac_path)
    
    window_features_dict = _calculate_window_features(PathsConfig.accelerometer_data_path)
    features_with_tac_dict = _add_tac_reading_to_windowed_data(window_features_dict, full_tac_df)

    final_combined_pd = pd.concat(features_with_tac_dict.values())
    
    # convert the index  back
    final_combined_pd = final_combined_pd.reset_index().rename(columns={'window_start_time': DataConfig.acceleleration_time_column})
    final_combined_pd[DataConfig.acceleleration_time_column] = final_combined_pd[DataConfig.acceleleration_time_column].astype(np.int64)

    final_spark_df = spark.createDataFrame(final_combined_pd)

    final_spark_df.write.parquet(
        PathsConfig.accelerometer_with_tac_parquet_path,
        mode="overwrite",
        partitionBy=DataConfig.partition_column
    )

    logger.info(f"Written labeled feature data to: {PathsConfig.accelerometer_with_tac_parquet_path}")

    spark.stop()

if __name__ == "__main__":
    main()