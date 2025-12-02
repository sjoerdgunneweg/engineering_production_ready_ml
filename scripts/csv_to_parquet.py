import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from configs.configs import DataConfig, PathsConfig

def accelerometer_data_to_parquet(spark, input_path, output_path):
    """
    Reads the single Accelerometer CSV file, converts the Unix timestamp, 
    and writes to Parquet, partitioned by 'pid'.
    """
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    df_processed = df.withColumn(
        "readable_time", 
        F.from_unixtime(F.col(DataConfig.acceleleration_time_column))
    ).withColumn(
        "measurement_date",
        F.to_date(F.from_unixtime(F.col(DataConfig.acceleleration_time_column)))
    )

    df_processed.write.parquet(
        output_path, 
        mode="overwrite", 
        partitionBy=DataConfig.partition_column
    )
    
def clean_tac_data_to_parquet(spark, input_path, output_path):
    """
    Reads multiple TAC CSV files from a folder, derives 'pid' from filename,
    converts the Unix timestamp, and writes to Parquet, partitioned by 'pid'.
    """
    
    # Load all CSVs in the folder
    df = spark.read.csv(
        os.path.join(input_path, "*.csv"), 
        header=True, 
        inferSchema=True
    )
    
    # extract the person_id from the full input file path, and add as new column
    df_with_id = df.withColumn(
        DataConfig.partition_column,
        F.regexp_extract(F.input_file_name(), r'([A-Z]{2}\d{4})', 1) # regex finding pid pattern like 'AB1234'
    )
    
    # convert unix timestamp
    df_processed = df_with_id.withColumn(
        "readable_time", 
        F.from_unixtime(F.col(DataConfig.tac_time_column))
    ).withColumn(
        "measurement_date",
        F.to_date(F.from_unixtime(F.col(DataConfig.tac_time_column)))
    )

    df_processed.write.parquet(
        output_path, 
        mode="overwrite", 
        partitionBy=DataConfig.partition_column
    )
    

def main():
    spark = SparkSession.builder.master("local[2]").getOrCreate()
    
    accelerometer_data_to_parquet(spark, PathsConfig.accelerometer_data_path, PathsConfig.accelerometer_parquet_path)    
    clean_tac_data_to_parquet(spark, PathsConfig.clean_tac_path, PathsConfig.tac_parquet_path)
    
    spark.stop()
    print("CSV to Parquet conversion completed.")


if __name__ == "__main__":
    main()
