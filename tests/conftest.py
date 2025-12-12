import os
import pytest
from pyspark.sql import SparkSession

from configs.configs import run_config

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1" # to resolve timezone warning in pytest with spark


@pytest.fixture
def spark_fixture():
    spark = SparkSession.builder.master(run_config.spark_master_url).getOrCreate()
    yield spark
    