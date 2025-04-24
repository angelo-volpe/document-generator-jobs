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
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    "generate_document_samples",
    default_args=default_args,
    description="DAG to generate document samples",
    schedule_interval=None,
    catchup=False,
) as dag:

    generate_samples = DockerOperator(
        task_id="generate_samples",
        image="document-generator-jobs:latest",
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode="host",
        mounts=[
            Mount(
                source="/Users/volpea/Documents/projects/document-generator-job/data",
                target="/app/data",
                type="bind",
            )
        ],
        mount_tmp_dir=False,
        command="--job_name sampling"
        + " {{ '--document_id ' + dag_run.conf['document_id'] + ' --num_samples ' + dag_run.conf['num_samples']  + ' --publish'}}",
    )
