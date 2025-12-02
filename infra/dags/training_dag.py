import pendulum

from airflow import DAG
from docker.types import Mount
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator

from datetime import datetime
from datetime import timedelta

local_timezone = pendulum.timezone("Europe/Amsterdam")

default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG("nova_training", 
          default_args=default_args, 
          schedule="59 23 * * *", # every day at 23:59 --> @daily starts at 24:00
          start_date=datetime(2025, 11, 28, tzinfo=local_timezone), 
          catchup=False) 

preprocessing_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py --preprocess",
    image="nova:latest",
    network_mode="infra_default",
    task_id="preprocessing",
    mounts=[
        Mount(source="/Users/sjoerdgunneweg/Documents/MSc_AI/EngineeringProdReadyMLAI/Assignment_5/uva-mlprod-code/data", target="/nova/data", type="bind") # TODO this is my local filepath still
    ],
    dag=dag,
)

feat_eng_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py --feat-eng",
    image="nova:latest",
    network_mode="infra_default",  # note: In order services to communicate each other, we add them to the same network. You do not see in docker-compose.yaml, b/c by default it puts all the services under one network called "infra_default".
    task_id="feature_engineering",
    mounts=[
        Mount(source="/Users/sjoerdgunneweg/Documents/MSc_AI/EngineeringProdReadyMLAI/Assignment_5/uva-mlprod-code/data", target="/nova/data", type="bind") # TODO this is my local filepath still
    ],
    dag=dag,
)

model_training_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_training.py --training",
    image="nova:latest",
    network_mode="infra_default",
    task_id="training",
    mounts=[
        Mount(source="/Users/sjoerdgunneweg/Documents/MSc_AI/EngineeringProdReadyMLAI/Assignment_5/uva-mlprod-code/data", target="/nova/data", type="bind") # TODO this is my local filepath still
    ],
    dag=dag,
)

model_reloading_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main_api.py --reload",
    image="nova:latest",
    network_mode="infra_default",
    task_id="model_reloading",
    mounts=[
        Mount(source="/Users/sjoerdgunneweg/Documents/MSc_AI/EngineeringProdReadyMLAI/Assignment_5/uva-mlprod-code/data", target="/nova/data", type="bind") # TODO this is my local filepath still
    ],
    dag=dag,
)

(
    preprocessing_task
    >> feat_eng_task
    >> model_training_task
    >> model_reloading_task
)
