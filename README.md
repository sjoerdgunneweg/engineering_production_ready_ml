# Engineering Production Ready ML/AI Project
## Project Overview
TODO


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

Data shoud also first be downloaded and put in /data. The data structure will then look like: ... TODO,  Info about data, clean is the one i need and the all accelerometer data as well
first convert data to parquet using the csv_to parquet script

```bash 
python scripts/prepare_training_data.py
```

## End-to-End Operation (Demo)

TODO


## Implementation
