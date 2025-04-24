from ..logging_config import logger
from .data import get_sample_data
from .model import get_annotations, train_test_split


def run_sample_preprocessing(document_id: int):
    logger.info(f"Running sample preprocessing for document: {document_id}")

    logger.info(f"Retrieving samples for document: {document_id}")
    images, box_labels = get_sample_data(document_id)
    logger.info(f"Retrieved {len(images)} samples")

    logger.info("Preprocess sample")
    images = get_annotations(images, box_labels)

    logger.info("Train Validation split")
    train_test_split(document_id, images, split_percentage=0.9)

    logger.info("Sample preprocessing completed")
