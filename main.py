import argparse
import logging
import datetime
from pathlib import Path

from jobs.handwritten_dataset_preprocessing.preprocessing import run_hw_preprocessing
from jobs.handwritten_dataset_preprocessing_emnist.preprocessing import (
    run_hw_preprocessing_emnist,
)
from jobs.sample_generation.generate_sample import run_sampling
from jobs.sample_preprocessing.preprocessing import run_sample_preprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_name",
        type=str,
        help='must be one of: "hw_preprocessing", "sampling", "sample_preprocessing"',
        required=True,
    )
    parser.add_argument("--document_id", type=int, required=False)
    parser.add_argument("--num_samples", type=int, required=False)
    parser.add_argument("--config_file_path", type=str, required=False)
    parser.add_argument("--output_base_path", type=str, required=False)
    parser.add_argument("--publish", action="store_true")

    args = parser.parse_args()

    if args.job_name == "hw_preprocessing":
        run_hw_preprocessing()
    elif args.job_name == "hw_preprocessing_emnist":
        run_hw_preprocessing_emnist(dataset_type="train")
        run_hw_preprocessing_emnist(dataset_type="test")
    elif args.job_name == "sampling":
        num_samples_train = int(args.num_samples * 0.8)
        num_samples_test = args.num_samples - num_samples_train
        version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        job_args = {
            "document_id": args.document_id,
            "num_samples": args.num_samples,
            "version": version,
            "publish": args.publish,
        }
        if args.output_base_path:
            output_base_path = args.output_base_path
            job_args["output_base_path"] = Path(output_base_path)

        if args.config_file_path:
            config_path = args.config_file_path
            job_args["config_path"] = Path(config_path)

        run_sampling(
            **job_args,
            dataset_type="train",
        )
        run_sampling(
            **job_args,
            dataset_type="test",
        )
    elif args.job_name == "sample_preprocessing":
        run_sample_preprocessing(document_id=args.document_id)
    else:
        logging.error("job_name not valid")
