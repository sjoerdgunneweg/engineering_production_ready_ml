import pytest
from pyspark.sql import SparkSession

from configs.configs import run_config


@pytest.fixture
def spark_fixture():
    spark = SparkSession.builder.master(run_config.spark_master_url).getOrCreate()
    yield spark
