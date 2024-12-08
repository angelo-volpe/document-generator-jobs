import argparse
import logging
from jobs.handwritten_dataset_preprocessing.preprocessing import run_preprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, help='"preprocessing" or "sample"')
    args = parser.parse_args()

    if args.job_name == "preprocessing":
        run_preprocessing()
    elif args.job_name == "sample":
        pass
        #run_sample()
    else:
        logging.error("job_name not valid")