import argparse
import logging
from jobs.handwritten_dataset_preprocessing.preprocessing import run_hw_preprocessing
from jobs.sample_generation.generate_sample import run_sampling
from jobs.sample_preprocessing.preprocessing import run_sample_preprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, help='must be one of: "hw_preprocessing", "sampling", "sample_preprocessing"', required=True)
    parser.add_argument('--document_id', type=int, required=False)
    parser.add_argument('--num_samples', type=int, required=False)
    parser.add_argument('--publish', action="store_true")

    args = parser.parse_args()

    if args.job_name == "hw_preprocessing":
        run_hw_preprocessing()
    elif args.job_name == "sampling":
        run_sampling(document_id=args.document_id, num_samples=args.num_samples, publish=args.publish)
    elif args.job_name == "sample_preprocessing":
        run_sample_preprocessing(document_id=args.document_id)
    else:
        logging.error("job_name not valid")