
import mock
import pytest
import pandas as pd


from src.utils.mlflow_utils import create_mlflow_experiment_if_not_exist, create_mlflow_run_if_not_exists, get_latest_run_id, save_artifacts_to_mlflow, log_metrics_to_mlflow


@pytest.fixture
def mock_mlflow_experiment():
    mock_experiment = mock.MagicMock()
    mock_experiment.experiment_id = "123"
    return mock_experiment



def test_create_mlflow_experiment_if_not_exist():
    with mock.patch("mlflow.get_experiment_by_name", return_value=None):
        with mock.patch("mlflow.create_experiment") as mock_create_experiment:
            create_mlflow_experiment_if_not_exist()
            mock_create_experiment.assert_called_once()

def test_create_mlflow_experiment_if_exists():
    with mock.patch("mlflow.get_experiment_by_name", return_value=mock.MagicMock()):
        with mock.patch("mlflow.create_experiment") as mock_create_experiment:
            create_mlflow_experiment_if_not_exist()
            mock_create_experiment.assert_not_called()

def test_create_mlflow_run_if_not_exists():
    with mock.patch("mlflow.get_experiment_by_name", return_value=mock.MagicMock(experiment_id="123")):
        with mock.patch("mlflow.search_runs", return_value=pd.DataFrame()):
            with mock.patch("mlflow.start_run") as mock_start_run:
                create_mlflow_run_if_not_exists("test_run")
                mock_start_run.assert_called_once()

def test_create_mlflow_run_if_exists():
    with mock.patch("mlflow.get_experiment_by_name", return_value=mock.MagicMock(experiment_id="123")):
        with mock.patch("mlflow.search_runs", return_value=pd.DataFrame({"run_id": ["1"]})):
            with mock.patch("mlflow.start_run") as mock_start_run:
                create_mlflow_run_if_not_exists("test_run")
                mock_start_run.assert_not_called()

def test_get_latest_run_id_no_experiment():
    """
    Tests that error is raised if experiment does not exist.
    """
    
    with mock.patch("mlflow.get_experiment_by_name", return_value=None):
        with pytest.raises(RuntimeError, match="does not exist in MLFlow"):
            get_latest_run_id("run")

def test_get_latest_run_id_no_runs(mock_mlflow_experiment):
    """
    Tests that error is raised if no runs exist for the given run name.
    """
    with mock.patch("mlflow.get_experiment_by_name", return_value=mock_mlflow_experiment):
        with mock.patch("mlflow.search_runs", return_value=pd.DataFrame()):
            with pytest.raises(RuntimeError, match="is not found in MLFlow"):
                get_latest_run_id("run")

def test_save_artifacts_to_mlflow():
    pass

def test_log_metrics_to_mlflow():
    pass