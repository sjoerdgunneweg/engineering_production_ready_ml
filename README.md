# Engineering Production Ready ML/AI Project

## Core ML/AI Task and Dataset

* **Dataset Chosen:** [Bar Crawl: Detecting Heavy Drinking](https://archive.ics.uci.edu/dataset/515/bar+crawl+detecting+heavy+drinking)
* **Task Type:** Classification
* **Specific Task:** Predicting whether person is intoxicated given their phone accelerometer data.

## System Architecture & Components

| Component                      | Tool/Library Chosen                               |
|-------------------------------|----------------------------------------------------|
| Programming Language          | Python                                             | 
| Distributed Data Processing   | PySpark              | 
| Containerization              | Docker                                             | 
| Integration Testing Setup     | Docker Compose                                     |
| Batch Pipeline Orchestrator   | Airflow             | 
| API Framework                 | Flask                                  | 
| API Deployment                | Docker Compose                      | 
| Model/Experiment Tracking     | MLflow                                             | 
| Telemetry Database            | Prometheus                                         | 
| Telemetry Dashboard           | Grafana                                            | 
| Alert Manager                 | [Prometheus AlertManager / Grafana AlertManager]  | 
| Testing Framework             | Pytest / Unittest                                  | 


## Setup and Installation
### Local Setup


```bash
pip install -r requirements.txt
export PYTHONPATH=.
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
```

The [dataset](https://archive.ics.uci.edu/dataset/515/bar+crawl+detecting+heavy+drinking) should first be downloaded and put in /data. The data structure will then look as such:

```text
data
├── bar+crawl+detecting+heavy+drinking
│   ├── data
│   │   ├── all_accelerometer_data_pids_13.csv
│   │   ├── clean_tac
│   │   │   ├── BK7610_clean_TAC.csv
│   │   │   ├── BU4707_clean_TAC.csv
│   │   │   ├── CC6740_clean_TAC.csv
│   │   │   ├── DC6359_clean_TAC.csv
│   │   │   ├── DK3500_clean_TAC.csv 
│   │   │   ├── HV0618_clean_TAC.csv
│   │   │   ├── JB3156_clean_TAC.csv
│   │   │   ├── JR8022_clean_TAC.csv
│   │   │   ├── MC7070_clean_TAC.csv
│   │   │   ├── MJ8002_clean_TAC.csv
│   │   │   ├── PC6771_clean_TAC.csv
│   │   │   ├── SA0297_clean_TAC.csv
│   │   │   └── SF3079_clean_TAC.csv
│   │   ├── phone_types.csv
│   │   ├── pids.txt
│   │   ├── raw_tac
│   │   │   ├── BK7610 CAM Results.xlsx
│   │   │   ├── BU4707 CAM results.xlsx
│   │   │   ├── CC6740 CAM Results.xlsx
│   │   │   ├── DC6359 CAM Results.xlsx
│   │   │   ├── DK3500 CAM Results.xlsx
│   │   │   ├── HV0618 CAM Results.xlsx
│   │   │   ├── JB3156 CAM Results.xlsx
│   │   │   ├── JR8022 CAM results.xlsx
│   │   │   ├── MC7070 CAM Results.xlsx
│   │   │   ├── MJ8002 CAM Results.xlsx
│   │   │   ├── PC6771 CAM Results.xlsx
│   │   │   ├── SA0297 CAM Results.xlsx
│   │   │   └── SF3079 CAM Results.xlsx
│   │   └── README.txt
│   └── Readme.txt
└── telemetry
    ├── data_dist.json
    └── live_data_dist.json
```

Next, prepare the training data and convert to parquet using:

```bash 
python scripts/prepare_training_data.py
```

## Demo

#### 1. Start by running
```bash
make up
```
#### 2. Once Airflow is up, trigger dag

Open [Airflow](http://localhost:4242/dags/alcoholerometer_training) and trigger the dag by hitting the "Trigger" button in the right corner

#### 3. Once dag has been triggered, and training is finished run:

```bash
docker compose -f infra/docker-compose.yaml up
```

#### 4. Api calling 

In a new terminal window, run:

```bash
bash scripts/api_request_example.sh
```

#### 5. Opening grafana
Open the [Grafana dashboard](http://localhost:3000/dashboards) by clicking on the dashboard called 'Alcoholerometer'


### Testing

1. Start by running 
```bash
make up
```

2. Once Airflow is up, trigger dag

3. In a new terminal window run:
```bash
make test
```

