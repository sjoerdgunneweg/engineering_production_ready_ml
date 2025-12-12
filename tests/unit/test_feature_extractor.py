import os
import tempfile
from mock import mock


import pytest


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

def test__get_is_intoxicated():
    pass

def get_energy():
    pass

def get_mean_energy():
    pass

def get_magnitude():
    pass
    
def get_mean_magnitude():
    pass

