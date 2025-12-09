import os
import pandas as pd
import numpy as np
import re
from pyspark.sql import SparkSession
from typing import Dict, List
from pyspark.sql.types import StructType, StructField, StringType, LongType, FloatType

from configs.configs import DataConfig, PathsConfig

# --- Define the final schema for the Parquet write ---
FINAL_SCHEMA = StructType([
    StructField(DataConfig.partition_column, StringType(), True),
    StructField(DataConfig.acceleleration_time_column, LongType(), True), # Window start time in Unix seconds
    StructField(DataConfig.tac_reading_column, FloatType(), True),
    StructField("x", FloatType(), True),
    StructField("y", FloatType(), True),
    StructField("z", FloatType(), True),
    # Include all other features calculated in create_sample_feature_frames here.
])

# Get the definitive column order from the schema for Pandas selection later
FINAL_COLUMN_ORDER: List[str] = [field.name for field in FINAL_SCHEMA.fields]
# ---------------------------------------------------


def _consolidate_full_tac_timeseries(input_path: str) -> pd.DataFrame:
    """
    Reads all raw TAC CSVs, extracts PID, and consolidates the full, 
    unaggregated time series into one Pandas DataFrame.
    
    Returns a single, time-sorted Pandas DataFrame of TAC readings.
    """
    if not os.path.isdir(input_path):
        print(f"ERROR: TAC data directory not found at: {input_path}")
        return pd.DataFrame()

    print(f"Consolidating FULL TAC time series data from {input_path} with Pandas...")
    
    tac_files = [
        os.path.join(input_path, entry.name) 
        for entry in os.scandir(input_path) 
        if entry.is_file() and entry.name.endswith('.csv')
    ]
    
    all_tac_data: List[pd.DataFrame] = []
    pid_regex = r'([A-Z]{2}\d{4})'

    for file_path in tac_files:
        file_name = os.path.basename(file_path)
        match = re.search(pid_regex, file_name)
        
        if not match: continue
        
        pid = match.group(1)
        
        try:
            tac_df = pd.read_csv(file_path, usecols=[DataConfig.tac_time_column, DataConfig.tac_reading_column])
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        tac_df[DataConfig.partition_column] = pid
        # Convert timestamp (Unix seconds) to datetime objects (Verified to be 's')
        tac_df[DataConfig.tac_time_column] = pd.to_datetime(tac_df[DataConfig.tac_time_column], unit='s') 
        
        all_tac_data.append(tac_df)

    if not all_tac_data:
        print("WARNING: No TAC CSV files found or processed.")
        return pd.DataFrame()
        
    tac_combined_df = pd.concat(all_tac_data, ignore_index=True)
    return tac_combined_df.sort_values([DataConfig.partition_column, DataConfig.tac_time_column])


def add_tac_label_to_windowed_data(
    feature_frames_dict: Dict[str, pd.DataFrame], 
    spark: SparkSession
) -> Dict[str, pd.DataFrame]:
    """
    Adds the closest previous TAC_Reading label to each feature-engineered time window 
    using the optimized Pandas merge_asof operation.
    """
    
    full_tac_timeseries_pd = _consolidate_full_tac_timeseries(PathsConfig.clean_tac_path)
    
    if full_tac_timeseries_pd.empty:
        print("FATAL: Cannot label data because TAC time series could not be loaded.")
        return feature_frames_dict

    # Group TAC data by PID for fast extraction inside the loop
    tac_pd_by_pid: Dict[str, pd.DataFrame] = {
        pid: df
        for pid, df in full_tac_timeseries_pd.groupby(DataConfig.partition_column)
    }

    labeled_frames: Dict[str, pd.DataFrame] = {}

    print("Applying TAC label alignment (merge_asof) to feature windows...")

    for pid, features_pd in feature_frames_dict.items():
        tac_pd_single_pid = tac_pd_by_pid.get(pid, pd.DataFrame())

        if tac_pd_single_pid.empty:
            print(f"   -> Skipping PID {pid}: No corresponding TAC data found.")
            labeled_frames[pid] = features_pd 
            continue
            
        # --- 1. PREPARATION: Move Index to Column and Convert Time ---
        # The index contains the time window key (e.g., 1493726840.0)
        # We must reset the index to make the time column available for the merge.
        features_pd_reset = features_pd.reset_index()
        
        # 2. Convert the window time column (assumed to be Unix seconds) to datetime objects
        # The index name from the create_sample_feature_frames is 'window_start_time'
        time_column_name = features_pd.index.names[0] 
        features_pd_reset['window_time_dt'] = pd.to_datetime(
            features_pd_reset[time_column_name], unit='s'
        )
        
        # NOTE: features_pd_reset MUST be sorted by the datetime column for merge_asof
        features_pd_sorted = features_pd_reset.sort_values('window_time_dt')

        # --- 3. The Time-Series Look-Back Join (Labeling) ---
        labeled_pd = pd.merge_asof(
            left=features_pd_sorted, 
            right=tac_pd_single_pid,
            left_on='window_time_dt', # Use the new datetime column
            right_on=DataConfig.tac_time_column,
            by=DataConfig.partition_column, 
            direction='backward'           
        )
        
        # --- 4. Cleanup and Restore Original Index ---
        
        # Drop temporary columns used for the merge (datetime column and TAC timestamp column)
        labeled_pd = labeled_pd.drop(
            columns=[DataConfig.tac_time_column, 'window_time_dt'], errors='ignore'
        )
        
        # Restore the index using the original time column name.
        labeled_pd = labeled_pd.set_index(time_column_name)

        # Ensure the TAC_Reading is float 
        labeled_pd[DataConfig.tac_reading_column] = labeled_pd[DataConfig.tac_reading_column].astype(float)
        
        labeled_frames[pid] = labeled_pd
        print(f"   -> Labeled PID {pid}. First 3 TAC readings: {labeled_pd[DataConfig.tac_reading_column].head(3).tolist()}")

    print("✅ TAC label assignment complete.")
    return labeled_frames


def create_sample_feature_frames(accel_path: str) -> Dict[str, pd.DataFrame]:
    """
    Simulates the feature engineering step: loads the raw data and groups it by 
    a simplified 10-second window, calculating the mean of x, y, z.
    The resulting DataFrame index is the window start time (in seconds).
    """
    print(f"--- Simulating Feature Engineering from: {accel_path} ---")
    
    # Load entire accelerometer data into Pandas
    raw_df = pd.read_csv(accel_path)
    
    # Convert time from milliseconds (ms) to seconds (s)
    raw_df['time_s'] = raw_df[DataConfig.acceleleration_time_column] / 1000.0
    
    # Create a 10-second window key (Unix seconds truncated to the nearest 10)
    WINDOW_SIZE = 10 
    raw_df['window_key'] = (raw_df['time_s'] // WINDOW_SIZE) * WINDOW_SIZE
    
    feature_frames_dict = {}
    
    for pid, group_df in raw_df.groupby(DataConfig.partition_column):
        
        # Group by the window key and calculate features (mean of x, y, z)
        feature_frame = group_df.groupby('window_key')[['x', 'y', 'z']].mean()
        
        # Add the PID back as a column for the merge_asof 'by' parameter
        feature_frame[DataConfig.partition_column] = pid
        
        # Set the index name, which will be used as the time column in the merge logic
        feature_frame.index.name = 'window_start_time' 
        
        feature_frames_dict[pid] = feature_frame
        
    print(f"--- Simulation Complete. Created {len(feature_frames_dict)} feature frames. ---")
    return feature_frames_dict


def main():
    # Configure Spark with high memory, as the initial CSV read into Pandas is still large
    spark = (
        SparkSession.builder
        .master("local[*]")
        .config("spark.driver.memory", "8g") 
        .getOrCreate()
    )
    
    # 1. Simulate the step that creates the windowed features
    feature_frames = create_sample_feature_frames(PathsConfig.accelerometer_data_path)
    
    # 2. Add the TAC label to the windowed features
    labeled_frames = add_tac_label_to_windowed_data(feature_frames, spark)
    
    # 3. CONSOLIDATE AND WRITE THE FINAL PARQUET FILE
    print("Consolidating labeled frames for final Parquet write...")
    
    if not labeled_frames:
        print("ERROR: No labeled frames to write.")
        spark.stop()
        return

    # Consolidate all labeled Pandas DataFrames into one large Pandas DataFrame
    final_combined_pd = pd.concat(labeled_frames.values())
    
    # Convert the index (window_start_time) back to a column named 'time' (LongType)
    final_combined_pd = final_combined_pd.reset_index().rename(
        columns={'window_start_time': DataConfig.acceleleration_time_column}
    )
    # Ensure the final time column is converted to the required LongType (integer seconds)
    final_combined_pd[DataConfig.acceleleration_time_column] = final_combined_pd[DataConfig.acceleleration_time_column].astype(np.int64)

    # Convert the large Pandas DF to Spark DF
    final_spark_df = spark.createDataFrame(final_combined_pd)

    # Write the final Parquet file (partitioned for performance)
    final_spark_df.write.parquet(
        PathsConfig.accelerometer_with_tac_parquet_path,
        mode="overwrite",
        partitionBy=DataConfig.partition_column
    )
    print(f"✅ FINAL SUCCESS: Labeled features written to {PathsConfig.accelerometer_with_tac_parquet_path}")

    spark.stop()

if __name__ == "__main__":
    main()