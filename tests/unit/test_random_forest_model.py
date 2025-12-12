import mock

import pandas as pd

from src.random_forest_model import RandomForestModel


def test_get_model():

    class MockModelConfig:
        max_depth = 5
        n_estimators = 100
        random_seed = 42

    with mock.patch('src.random_forest_model.ModelConfig', MockModelConfig):
        model = RandomForestModel._get_model()
        assert model.max_depth == MockModelConfig.max_depth 
        assert model.n_estimators == MockModelConfig.n_estimators  
        assert model.random_state == MockModelConfig.random_seed  


def test_model_get_x_y():
    cases = [
        {
            "data": pd.DataFrame({
                "pid": ["A", "B", "C"],
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "is_intoxicated": [0, 1, 0],
                "TAC_Reading": [0.05, 0.1, 0.03],
            }),
            "expected_x": pd.DataFrame({
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
            }),
            "expected_y": pd.Series([False, True, False], name="is_intoxicated"),
        },
        {
            "data": pd.DataFrame({
                "pid": ["D", "E"],
                "feature1": [7, 8],
                "feature2": [9, 10],
                "is_intoxicated": [1, 1],
                "TAC_Reading": [0.12, 0.15],
            }),
            "expected_x": pd.DataFrame({
                "feature1": [7, 8],
                "feature2": [9, 10],
            }),
            "expected_y": pd.Series([True, True], name="is_intoxicated"),
        },
    ]

    for case in cases:
        random_forest_model = RandomForestModel()
        out_x, out_y = random_forest_model._get_x_y(case["data"])
        pd.testing.assert_frame_equal(out_x, case["expected_x"])
        pd.testing.assert_series_equal(out_y, case["expected_y"])


def test_remove_non_training_features():
    data = pd.DataFrame({
        "pid": ["A", "B", "C"],
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "is_intoxicated": [0, 1, 0],
        "TAC_Reading": [0.05, 0.1, 0.03],
    })

    expected_data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "is_intoxicated": [0, 1, 0],
    })

    random_forest_model = RandomForestModel()
    out_data = random_forest_model._remove_non_training_features(data)
    pd.testing.assert_frame_equal(out_data, expected_data)