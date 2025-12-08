from pyspark.sql import DataFrame
from pyspark.sql import SparkSession


def read_parquet(parquet_path: str) -> DataFrame:
    spark = SparkSession.builder \
        .master("local[*]") \
        .getOrCreate()
    
    return spark.read.parquet(parquet_path)
