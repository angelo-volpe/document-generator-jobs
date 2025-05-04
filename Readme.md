### Dataset Source

EMNIST Dataset: https://www.nist.gov/itl/products-and-services/emnist-dataset

### Jobs
- <b>Handwritten Dataset Preprocessing</b>: read all images from handwritten dataset and preprocess them(adding alpha, crop to content)
- <b>Sample Generation</b>: read the document and the handwritten dataset and generate the sample. [Job Details](./jobs/sample_generation/README.md)

### How to run jobs

#### In local environment

Run Handwritten Dataset Preprocessing Job
```bash
python main.py --job_name hw_preprocessing_emnist
```

Run Sample Generation Job
```bash
python main.py --job_name sampling --document_id 3 --num_samples 10
```

Run Sample Preprocessing Job
```bash
python main.py --job_name sample_preprocessing --document_id 5
```

#### using Docker
```bash
docker build . -t document-generator-jobs:latest
docker run --rm \
           --mount type=bind,source=./data,target=/app/data \
           --env-file .env.dev \
           --name document-generator-jobs \
           document-generator-jobs:latest --job_name sampling --document_id 3 --num_samples 10
```

### Start Airflow
```
cd airflow
docker compose up airflow-init
docker compose up
```