from dataclasses import dataclass

@dataclass(init=False, frozen=True)
class PathsConfig:
    clean_tac_path: str = "data/bar+crawl+detecting+heavy+drinking/data/clean_tac/" # TODO fix to better data paths? now have data in data
    accelerometer_data_path: str = "data/bar+crawl+detecting+heavy+drinking/data/all_accelerometer_data_pids_13.csv"

    accelerometer_parquet_path: str = "data/accelerometer/"
    tac_parquet_path: str = "data/tac/"

@dataclass(init=False, frozen=True)
class DataConfig:
    acceleleration_time_column: str = "time"
    tac_time_column: str = "timestamp"
    partition_column: str = "pid"

    tac_time_column: str = "timestamp"
    

@dataclass(init=False, frozen=True)
class RunConfig:
    random_seed: int = 42

@dataclass(init=False, frozen=True)
class ModelConfig:
    learning_rate: float = 0.01
    num_epochs: int = 100
