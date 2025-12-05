from dataclasses import dataclass, field

@dataclass(init=False, frozen=True)
class PathsConfig:
    clean_tac_path: str = "data/bar+crawl+detecting+heavy+drinking/data/clean_tac/" # TODO fix to better data paths? now have data in data
    accelerometer_data_path: str = "data/bar+crawl+detecting+heavy+drinking/data/all_accelerometer_data_pids_13.csv"

    accelerometer_parquet_path: str = "data/accelerometer/"
    tac_parquet_path: str = "data/tac/"

    telemetry_training_data_path: str = "data/telemetry/data_dist.json" # TODO change to better path
    telemetry_live_data_path: str = "data/telemetry/live_data_dist.json" # TODO change to better path

@dataclass(init=False, frozen=True)
class DataConfig:
    acceleleration_time_column: str = "time"
    tac_time_column: str = "timestamp"
    partition_column: str = "pid"

    tac_time_column: str = "timestamp"

    person_ids: list[str] = field(default_factory=lambda: [
        "BK7610", "BU4707", "CC6740", "DC6359", "DK3500",
        "HV0618", "JB3156", "JR8022", "MC7070", "MJ8002",
        "PC6771", "SA0297", "SF3079"
    ])

    intoxication_threshold: float = 0.08  # TODO find source, but maybe also not in right config

@dataclass(frozen=True)
class _RunConfig:
    """
    This class should not be used directly without instantiation. Always use the run_config in this module.

    Allows config to be overridden by environment variables if needed.
    """
    app_name: str = "alcoholerometer"
    spark_master_url: str = "local[2]"
    mlflow_tracking_uri: str = "http://mlflow:8080" #"http://localhost:8080"
    experiment_name: str = "Alpha" # TODO can i set this to something else?
    run_name: str = "Run_234" # TODO is this needed?
    random_seed: int = 42
    sample_rate: float = 1.0 

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


@dataclass(init=False, frozen=True)
class ModelConfig:
    random_seed: int = 42
    max_depth: int = 10
    n_estimators: int = 100


run_config = _RunConfig()