from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import timedelta
from docker.types import Mount


default_args = {
    "owner": "fox",
    "depends_on_past": False,
    "email": ["angelovolpe95@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    "faster_rcnn_fine_tuning",
    default_args=default_args,
    description="DAG to fine tune faster-rcnn model for specific document",
    schedule_interval=None,
    catchup=False,
) as dag:

    training = DockerOperator(
        task_id="train_model",
        image="document-generator-text-detector:latest",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="host",
        shm_size=2 * 1024 * 1024 * 1024,
        environment={"MLFLOW_TRACKING_URI": "http://host.docker.internal:5000"},
        mounts=[
            Mount(
                source="/Users/volpea/Documents/projects/document-generator-job/data/sampling",
                target="/app/data",
                type="bind",
            ),
        ],
        mount_tmp_dir=False,
        command="--document_id {{ dag_run.conf['document_id'] }} --input_base_path /app/data",
    )
