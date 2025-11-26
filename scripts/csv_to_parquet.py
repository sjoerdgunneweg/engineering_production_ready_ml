import argparse

import pandas as pd
from pyspark.sql import SparkSession


def csv_to_parquet(csv_file_path: str, parquet_file_path: str):
    raw_csv = pd.read_csv(csv_file_path)

    #TODO check of bepaalde dingen weg mogen uit de dataset


    spark = SparkSession.builder.appName("CSV to Parquet").getOrCreate()
    spark_df = spark.createDataFrame(raw_csv)
    
    spark_df.write.parquet(parquet_file_path, mode="overwrite", partitionBy=TODO) # TODO specify partition columns if needed


def main(args):
    csv_to_parquet(args.csv_file_path, args.parquet_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--parquet_file_path", type=str, required=True, help="Path to the output Parquet file")
    args = parser.parse_args()

    main(args)