import argparse
import logging
from jobs.handwritten_dataset_preprocessing.preprocessing import run_preprocessing
from jobs.sample_generation.generate_sample import run_sampling


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, help='"preprocessing" or "sampling"', required=True)
    parser.add_argument('--document_id', type=int, required=False)
    parser.add_argument('--num_samples', type=int, required=False)

    args = parser.parse_args()

    if args.job_name == "hw_preprocessing":
        run_preprocessing()
    elif args.job_name == "sampling":
        run_sampling(document_id=args.document_id, num_samples=args.num_samples)
    else:
        logging.error("job_name not valid")