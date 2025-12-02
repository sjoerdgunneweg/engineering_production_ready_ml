import argparse

import pandas as pd
from pyspark.sql import SparkSession


def csv_to_parquet(csv_file_path: str, parquet_file_path: str):
    spark = SparkSession.builder.appName("CSV to Parquet").getOrCreate()
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    
    df = df.toPandas()

    print(df.head())


def main(args):
    csv_to_parquet(args.csv_file_path, args.parquet_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--parquet_file_path", type=str, required=True, help="Path to the output Parquet file")
    args = parser.parse_args()

    main(args)