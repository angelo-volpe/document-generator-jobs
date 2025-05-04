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
    "paddle_ocr_fine_tuning",
    default_args=default_args,
    description="DAG to fine tune Paddle OCR model for specific document",
    schedule_interval=None,
    catchup=False,
) as dag:

    training = DockerOperator(
        task_id="train_model",
        image="paddle-ocr-document-training:latest",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="host",
        shm_size=2 * 1024 * 1024 * 1024,
        mounts=[
            Mount(
                source="/Users/volpea/Documents/projects/document-generator-job/data",
                target="/data",
                type="bind",
            ),
            Mount(
                source="/Users/volpea/Documents/projects/PaddleOCR/models",
                target="/models",
                type="bind",
            ),
        ],
        mount_tmp_dir=False,
        command="./document_app/train_new_model.sh /data/fine_tuning_dataset {{ dag_run.conf['document_id'] }}",
    )
