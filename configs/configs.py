from dataclasses import dataclass

@dataclass(init=False, frozen=True)
class PathsConfig:
    clean_tac_path: str = "data/bar+crawl+detecting+heavy+drinking/data/clean_tac/"
    accelerometer_data_path: str = "data/bar+crawl+detecting+heavy+drinking/data/all_accelerometer_data_pids_13.csv"

    accelerometer_parquet_path: str = "data/accelerometer/"
    tac_parquet_path: str = "data/tac/"
    accelerometer_with_tac_parquet_path: str = "data/accelerometer_with_tac/"

    telemetry_training_data_path: str = "data/telemetry/data_dist.json" 
    telemetry_live_data_path: str = "data/telemetry/live_data_dist.json" 

    preprocessing_data_path: str = "data/processed/preprocessed_data.parquet"
    features_data_path: str = "data/processed/features_data.parquet"

    mean_energy_file_name: str = "mean_energy.pkl"
    std_energy_file_name: str = "std_energy.pkl"
    mean_magnitude_file_name: str = "mean_magnitude.pkl"
    std_magnitude_file_name: str = "std_magnitude.pkl"

@dataclass(init=False, frozen=True)
class DataConfig:
    acceleleration_time_column: str = "time"
    tac_time_column: str = "timestamp"
    partition_column: str = "pid"
    tac_reading_column = "TAC_Reading"

    window_size_seconds: int = 10
    pid_regex_pattern: str = r'([A-Z]{2}\d{4})' # regex finding pid pattern like 'AB1234'

    window_start_index_name: str = "window_start_time" # TODO think of better name
    datetime_column: str = "window_time_datetime"
    time_in_seconds_column: str = "window_time_seconds"
    window_key_column: str = "window_key"
    
    

@dataclass(frozen=True)
class _RunConfig:
    """
    This class should not be used directly without instantiation. Always use the run_config in this module.

    Allows config to be overridden by environment variables if needed.
    """
    app_name: str = "alcoholerometer"
    spark_master_url: str = "local[2]"
    mlflow_tracking_uri: str = "http://mlflow:8080" #"http://localhost:8080"
    experiment_name: str = "Alcoholerometer_Experiment"
    run_name: str = "alcoholerometer_random_forest_run"
    random_seed: int = 42
    sample_rate: float = 1.0 
    num_folds: int = 5
    intoxication_threshold: float = 0.08  
    metrics_to_log: tuple[str, ...] = ('test_precision', 'train_precision', 'test_recall', 'train_recall', 'test_f1', 'train_f1', 'test_accuracy', 'train_accuracy')
    
@dataclass(frozen=True)
class TelemetryConfig: # TODO alter to my style
    # The number of instances that should be in the live distribution.
    # As we will need a distribution that represents the "recent" status, we will need to form a distribution from
    # the "latest" data that the application has received. For this system, we define "latest" as the num_instances_for_live_dist
    # instances that the app received.
    # Therefore, we take the latest num_instances_for_live_dist instances as the live distribution for the
    # calculation of PSI.
    #
    # Example:
    #       num_instances_for_live_dist = 2
    #
    #   Call Time,  row id,     value
    #    10:30        1           0.1
    #    10:31        2           0.2
    #    11:32        3           0.3
    #
    # Live/"latest" distribution: [0.2, 0.3]
    #
    num_instances_for_live_dist: int = 3
    epsilon: float = 1 / 1e100
    push_gateway_uri: str = "http://prometheus_push_gateway:9091" # "http://localhost:9091"
    targets: tuple[str, ...] = ("is_intoxicated",)

@dataclass(init=False, frozen=True)
class ModelConfig:
    random_seed: int = 42
    max_depth: int = 10
    n_estimators: int = 100
    model_name: str = "alcoholerometer_random_forest"

run_config = _RunConfig()