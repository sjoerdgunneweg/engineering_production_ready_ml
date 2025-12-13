import tempfile

import mock

from pyspark.testing.utils import assertDataFrameEqual


from src.data.data_preprocessing import get_preprocessed_data


def test_get_preprocessed_data(spark_fixture):
    
    data = spark_fixture.createDataFrame(
        [
            {"pid": "A", "time": 1, "x": 0.1, "y": 0.2, "z": 0.3, "TAC_Reading": 0.4},
            {"pid": "B", "time": 2, "x": 0.4, "y": 0.5, "z": 0.6, "TAC_Reading": 0.7},
            {"pid": "C", "time": 3, "x": 0.7, "y": 0.8, "z": 0.9, "TAC_Reading": 1.0},
            {"pid": "D", "time": 4, "x": 1.0, "y": 1.1, "z": 1.2, "TAC_Reading": 1.3},
        ],
    )
    expected = spark_fixture.createDataFrame(
        [
            {"pid": "A", "time": 1, "x": 0.1, "y": 0.2, "z": 0.3, "TAC_Reading": 0.4},
            {"pid": "B", "time": 2, "x": 0.4, "y": 0.5, "z": 0.6, "TAC_Reading": 0.7},
            {"pid": "C", "time": 3, "x": 0.7, "y": 0.8, "z": 0.9, "TAC_Reading": 1.0},
            {"pid": "D", "time": 4, "x": 1.0, "y": 1.1, "z": 1.2, "TAC_Reading": 1.3},
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        class MockPathsConfig:
                accelerometer_with_tac_parquet_path = tmpdir + "/donono"

        with mock.patch('src.data.data_preprocessing.PathsConfig', MockPathsConfig):
            data.write.parquet(MockPathsConfig.accelerometer_with_tac_parquet_path, mode="overwrite") 

            out = get_preprocessed_data(spark_fixture)

            assertDataFrameEqual(out, expected, ignoreColumnOrder=True)   
    