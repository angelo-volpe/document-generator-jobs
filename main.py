import argparse
import logging
from jobs.handwritten_dataset_preprocessing.preprocessing import run_preprocessing
from jobs.sample_generation.generate_sample import run_sampling


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, help='"preprocessing" or "sampling"')
    args = parser.parse_args()

    if args.job_name == "hw_preprocessing":
        run_preprocessing()
    elif args.job_name == "sampling":
        run_sampling()
    else:
        logging.error("job_name not valid")