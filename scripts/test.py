import os
import pandas as pd
import numpy as np
import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from typing import Dict, List

from pyspark.sql.types import StructType, StructField, StringType, LongType, FloatType
from configs.configs import DataConfig, PathsConfig



class PathsConfig:
    # INPUT CSV paths
    accelerometer_data_path = "data/bar+crawl+detecting+heavy+drinking/data/all_accelerometer_data_pids_13.csv"
    clean_tac_path = "data/bar+crawl+detecting+heavy+drinking/data/clean_tac" 
    
    # FINAL OUTPUT Parquet path
    labeled_data_parquet_path = "output/labeled_training_data_preprocessed.parquet"

FINAL_SCHEMA = StructType([
    StructField(DataConfig.partition_column, StringType(), True),
    StructField(DataConfig.acceleleration_time_column, LongType(), True),
    StructField(DataConfig.tac_reading_column, FloatType(), True),
    StructField("x", FloatType(), True),
    StructField("y", FloatType(), True),
    StructField("z", FloatType(), True),
])

FINAL_COLUMN_ORDER: List[str] = [field.name for field in FINAL_SCHEMA.fields]
def _consolidate_and_clean_tac_data(input_path: str) -> pd.DataFrame:
    """
    Reads multiple TAC CSV files using Pandas, extracts PID from the filename 
    using pure Python regex, and consolidates the data.
    
    Returns a single, time-sorted Pandas DataFrame of TAC readings.
    """
    print("Consolidating TAC data with Pandas...")
    
    tac_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.csv')]
    all_tac_data: List[pd.DataFrame] = []
    pid_regex = r'([A-Z]{2}\d{4})'

    for file_path in tac_files:
        file_name = os.path.basename(file_path)
        
        match = re.search(pid_regex, file_name)
        
        if not match:
            continue
        
        pid = match.group(1)
        
        try:
            tac_df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        tac_df[DataConfig.partition_column] = pid
        # Convert timestamp to datetime objects (assuming unit='ms')
        tac_df[DataConfig.tac_time_column] = pd.to_datetime(tac_df[DataConfig.tac_time_column], unit='ms') 
        all_tac_data.append(tac_df)

    if not all_tac_data:
        return pd.DataFrame()
        
    tac_combined_df = pd.concat(all_tac_data, ignore_index=True)
    return tac_combined_df[[
        DataConfig.partition_column, 
        DataConfig.tac_time_column, 
        DataConfig.tac_reading_column
    ]].sort_values(DataConfig.tac_time_column)


def accelerometer_data_to_parquet(spark: SparkSession, accel_input_path: str, tac_data_path: str, output_path: str) -> None:
    """
    Reads Accelerometer CSV, performs the per-PID Pandas TAC join (label creation), 
    and writes the final labeled data to Parquet incrementally to avoid OOM errors.
    """
    print(f"Reading Accelerometer CSV from {accel_input_path} and TAC data from {tac_data_path}...")
    
    # 1. Prepare TAC data
    tac_combined_pd = _consolidate_and_clean_tac_data(tac_data_path)
    # Group TAC data by PID for fast access during the loop
    tac_pd_by_pid: Dict[str, pd.DataFrame] = {pid: df for pid, df in tac_combined_pd.groupby(DataConfig.partition_column)}

    # 2. Read Accelerometer Data (large file)
    accel_pd_all = pd.read_csv(accel_input_path)
    # Prepare accelerometer time column
    accel_pd_all[DataConfig.acceleleration_time_column] = pd.to_datetime(accel_pd_all[DataConfig.acceleleration_time_column], unit='ms')
    
    # 3. Incremental Processing Loop
    first_write = True
    print("Applying CRITICAL 'Add TAC Step' (Pandas merge_asof) and writing incrementally...")
    
    for pid, accel_pd_single_pid in accel_pd_all.groupby(DataConfig.partition_column):
        
        tac_pd_single_pid = tac_pd_by_pid.get(pid, pd.DataFrame())

        if tac_pd_single_pid.empty:
            print(f"   -> Skipping PID {pid}: No corresponding TAC data found.")
            continue
        
        # --- 3a. The time-series look-back join (Label Creation) ---
        labeled_pd = pd.merge_asof(
            left=accel_pd_single_pid.sort_values(DataConfig.acceleleration_time_column), 
            right=tac_pd_single_pid,
            left_on=DataConfig.acceleleration_time_column,
            right_on=DataConfig.tac_time_column,
            by=DataConfig.partition_column,  
            direction='backward'           
        )
        
        # 3b. Final cleaning and type preparation for Spark
        labeled_pd = labeled_pd.drop(columns=[DataConfig.tac_time_column], errors='ignore')
        labeled_pd[DataConfig.acceleleration_time_column] = labeled_pd[DataConfig.acceleleration_time_column].astype(np.int64) // 10**3
        labeled_pd[DataConfig.tac_reading_column] = labeled_pd[DataConfig.tac_reading_column].astype(float)
        
        # 3c. Order columns explicitly to match schema
        # Filter the DataFrame to include only the required columns, in the correct order
        current_cols = set(labeled_pd.columns)
        ordered_cols = [col for col in FINAL_COLUMN_ORDER if col in current_cols]
        labeled_pd = labeled_pd[ordered_cols]
        
        # 4. Convert to Spark DataFrame and Write Incrementally
        labeled_spark_df = spark.createDataFrame(labeled_pd, schema=FINAL_SCHEMA)

        write_mode = "overwrite" if first_write else "append"
        
        labeled_spark_df.write.parquet(
            output_path, 
            mode=write_mode, 
            partitionBy=DataConfig.partition_column
        )
        
        first_write = False
        print(f"   -> Wrote PID {pid} to Parquet in '{write_mode}' mode.")

    print(f"✅ Pre-enrichment step complete. Final data written to {output_path}")


def main():
    # Configure Spark for increased driver memory to handle the large Pandas read/write steps
    spark = (
        SparkSession.builder
        .master("local[*]")
        .config("spark.driver.memory", "8g") # Set to 8GB as this resolved the OOM
        .getOrCreate()
    )
    
    # Run the single, robust enrichment function
    accelerometer_data_to_parquet(
        spark, 
        PathsConfig.accelerometer_data_path, 
        PathsConfig.clean_tac_path, 
        PathsConfig.labeled_data_parquet_path
    )    
    
    spark.stop()

if __name__ == "__main__":
    main()