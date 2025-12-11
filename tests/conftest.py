import pytest
from pyspark.sql import SparkSession


@pytest.fixture
def spark_fixture():
    spark = SparkSession.builder.master("local[2]").getOrCreate()
    yield spark
    spark.stop() # TODO is this needed? i added this myself