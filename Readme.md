### Dataset Source
https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset

### Jobs
- <b>Handwritten Dataset Preprocessing</b>: read all images from handwritten dataset and preprocess them(adding alpha, crop to content)
- <b>Sample Generation</b>: read the document and the handwritten dataset and generate the sample


### How to run jobs

#### In local environment

Run Handwritten Dataset Preprocessing Job
```
python main.py --job_name hw_preprocessing
```

Run Sample Generation Job
```
python main.py --job_name sampling --document_id 3 --num_samples 10
```

Run Sample Preprocessing Job
```
python main.py --job_name sample_preprocessing --document_id 5
```

#### using Docker
```
docker build . -t document-generator-jobs:latest
docker run --rm \
           --network host \
           --mount type=bind,source=./data,target=/app/data \
           --name document-generator-jobs \
           document-generator-jobs:latest --job_name sampling --document_id 3 --num_samples 10
```

### Start Airflow
```
cd airflow
docker compose up airflow-init
docker compose up
```