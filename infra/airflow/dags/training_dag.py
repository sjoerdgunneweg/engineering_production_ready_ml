
from airflow import DAG
from docker.types import Mount
from airflow.providers.docker.operators.docker import DockerOperator

from datetime import timedelta


default_args = {
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG("alcoholerometer_training", default_args=default_args) 

preprocessing_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main.py --preprocess", 
    image="alcoholerometer:latest",
    network_mode="infra_default",
    task_id="preprocessing",
    mounts=[
        Mount(source="data", target="/alcoholerometer/data")
    ],
    dag=dag,
)

feat_eng_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main.py --feat_eng",
    image="alcoholerometer:latest",
    network_mode="infra_default",  # note: In order services to communicate each other, we add them to the same network. You do not see in docker-compose.yaml, b/c by default it puts all the services under one network called "infra_default".
    task_id="feature_engineering",
    mounts=[
        Mount(source="data", target="/alcoholerometer/data")
    ],
    dag=dag,
)

model_training_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/main.py --training",
    image="alcoholerometer:latest",
    network_mode="infra_default",
    task_id="training",
    mounts=[
        Mount(source="data", target="/alcoholerometer/data")
    ],
    dag=dag,
)

model_reloading_task = DockerOperator(
    docker_url="unix://var/run/docker.sock",
    command="python src/api.py --reload",
    image="alcoholerometer:latest",
    network_mode="infra_default",
    task_id="model_reloading",
    mounts=[
        Mount(source="data", target="/alcoholerometer/data")
    ],
    dag=dag,
)

(
    preprocessing_task
    >> feat_eng_task
    >> model_training_task
    >> model_reloading_task
)
