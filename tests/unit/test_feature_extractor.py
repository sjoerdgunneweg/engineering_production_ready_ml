import os
import tempfile
from mock import mock

from pyspark.testing.utils import assertDataFrameEqual
from src.features.feature_extractor import FeatureExtractor, _FeatureExtractorData


def test_feature_extractor_data_save():
    class MockPathsConfig:
        mean_energy_file_name = "foo1"
        std_energy_file_name = "foo2"
        mean_magnitude_file_name = "foo3"
        std_magnitude_file_name = "foo4"

    feature_extractor_data = _FeatureExtractorData()
    with mock.patch("src.features.feature_extractor.PathsConfig", MockPathsConfig()): # TODO shouldnt it be cofigs.PathsConfig ?
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_extractor_data.save(tmpdir)
            assert {
                MockPathsConfig.mean_energy_file_name,
                MockPathsConfig.std_energy_file_name,
                MockPathsConfig.mean_magnitude_file_name,
                MockPathsConfig.std_magnitude_file_name,
            } == set(os.listdir(tmpdir))

def test_feature_extractor_get_features_training_time():
    pass

def test_feature_extractor_get_features_training():
    pass

def test_feature_extractor_get_is_intoxicated(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"TAC_Reading": 0.03}, 
                    {"TAC_Reading": 0.09}, 
                    {"TAC_Reading": 0.05},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {"TAC_Reading": 0.03, "is_intoxicated": False}, 
                    {"TAC_Reading": 0.09, "is_intoxicated": True}, 
                    {"TAC_Reading": 0.05, "is_intoxicated": False},
                ]
            ),
        },
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"TAC_Reading": 0.00}, 
                    {"TAC_Reading": 0.02}, 
                    {"TAC_Reading": 0.01},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {"TAC_Reading": 0.00, "is_intoxicated": False}, 
                    {"TAC_Reading": 0.02, "is_intoxicated": False}, 
                    {"TAC_Reading": 0.01, "is_intoxicated": False},
                ]
            ),
        },
    ]

    class Mockrun_config:
        intoxication_threshold = 0.08

    for case in cases:
        with mock.patch("src.features.feature_extractor.run_config", Mockrun_config):
            feature_extractor = FeatureExtractor()
            out = feature_extractor._get_is_intoxicated(case["data"])
            assertDataFrameEqual(out, case["expected"],  ignoreColumnOrder=True)

def test_feature_extractor_get_energy(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"x": 1.0, "y": 2.0, "z": 3.0}, 
                    {"x": 4.0, "y": 5.0, "z": 6.0},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {"x": 1.0, "y": 2.0, "z": 3.0, "energy": 14.0}, 
                    {"x": 4.0, "y": 5.0, "z": 6.0, "energy": 77.0},
                ]
            ),
        },
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"x": 0.0, "y": 0.0, "z": 0.0}, 
                    {"x": 3.0, "y": 4.0, "z": 0.0},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "energy": 0.0}, 
                    {"x": 3.0, "y": 4.0, "z": 0.0, "energy": 25.0},
                ]
            ),
        },
    ] 

    for case in cases:
        feature_extractor = FeatureExtractor()
        out = feature_extractor._get_energy(case["data"])
        assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)

def test_feature_extractor_get_magnitude(spark_fixture):
    cases = [
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"x": 1.0, "y": 2.0, "z": 3.0, "energy": 14.0}, 
                    {"x": 4.0, "y": 5.0, "z": 6.0, "energy": 77.0},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {"x": 1.0, "y": 2.0, "z": 3.0, "energy": 14.0, "magnitude": 3.7416573868}, 
                    {"x": 4.0, "y": 5.0, "z": 6.0, "energy": 77.0, "magnitude": 8.7749643874},
                ]
            ),
        },
        {
            "data": spark_fixture.createDataFrame(
                [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "energy": 0.0}, 
                    {"x": 3.0, "y": 4.0, "z": 0.0, "energy": 25.0},
                ]
            ),
            "expected": spark_fixture.createDataFrame(
                [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "energy": 0.0, "magnitude": 0.0}, 
                    {"x": 3.0, "y": 4.0, "z": 0.0, "energy": 25.0, "magnitude": 5.0},
                ]
            ),
        },
    ]

    for case in cases:
        feature_extractor = FeatureExtractor()
        out = feature_extractor._get_magnitude(case["data"])
        assertDataFrameEqual(out, case["expected"], ignoreColumnOrder=True)

