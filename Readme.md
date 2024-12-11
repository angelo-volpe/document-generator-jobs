### Dataset Source
https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset

### Jobs
- <b>Handwritten Dataset Preprocessing</b>: read all images from handwritten dataset and preprocess them(adding alpha, crop to content)
- <b>Sample Generation</b>: read the document and the handwritten dataset and generate the sample


### How to run jobs

Run Handwritten Dataset Preprocessing Job
```
python main.py --job_name hw_preprocessing
```

Run Sample Generation Job
```
python main.py --job_name sampling
```