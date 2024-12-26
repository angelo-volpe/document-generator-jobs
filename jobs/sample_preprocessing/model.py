import json
import os
from pathlib import Path
from tqdm import tqdm

from ..logging_config import logger


def denormalise_box_coordinates(start_x_norm: float, start_y_norm: float, end_x_norm: float, end_y_norm: float, 
                                doc_width: int, doc_height: int):
    start_x = int(start_x_norm * doc_width)
    end_x = int(end_x_norm * doc_width)
    start_y = int(start_y_norm * doc_height)
    end_y = int(end_y_norm * doc_height)
    
    return start_x, start_y, end_x, end_y

def get_box_coords(start_x, start_y, end_x, end_y):
    return [[start_x, start_y], [end_x, start_y], [end_x, end_y], [start_x, end_y]]


def get_annotations(images, box_labels):
    sample_image = images[0]["image"]
    image_width = sample_image.width
    image_height = sample_image.height

    for sample in images:
        sample_labels = box_labels[box_labels["sample_document"] == sample["sample_id"]]
        boxes_annotation = sample_labels.apply(lambda x: 
                                            {"transcription": x["label"], 
                                             "points": get_box_coords(*denormalise_box_coordinates(x["start_x_norm"], x["start_y_norm"], 
                                                                                                   x["end_x_norm"], x["end_y_norm"], 
                                                                                                   image_width, image_height))}
                        , axis=1).values

        sample["annotations"] = json.dumps(list(boxes_annotation))

    return images


def train_test_split(document_id: int, images, split_percentage=0.9):
    num_samples = len(images)
    fine_tuning_dataset_path = Path(os.environ.get("FINE_TUNING_DATASET_PATH", "./data/fine_tuning_dataset/")) / f"document_{document_id}/"

    split = int(num_samples * split_percentage)
    train_samples = images[:split]
    val_samples = images[split:]

    logger.info(f"Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    logger.info(f"Writing train samples")
    train_annotations = []
    for sample in tqdm(train_samples):
        image_path = fine_tuning_dataset_path / f"train_images/"
        image_path.mkdir(parents=True, exist_ok=True)
        sample["image"].save(image_path / f"sample_{sample["sample_id"]}.png")

        train_annotation = str(image_path) + "\t" + sample["annotations"]
        train_annotations.append(train_annotation)

    logger.info(f"Writing validation samples")
    val_annotations = []
    for sample in tqdm(val_samples):
        image_path = fine_tuning_dataset_path / f"val_images/"
        image_path.mkdir(parents=True, exist_ok=True)
        sample["image"].save(image_path / f"sample_{sample["sample_id"]}.png")

        val_annotation = str(image_path) + "\t" + sample["annotations"]
        val_annotations.append(val_annotation)

    logger.info(f"Writing labels")
    with open(fine_tuning_dataset_path / 'train_labels.txt', "w") as file:
        file.write("\n".join(train_annotations))

    with open(fine_tuning_dataset_path / 'val_labels.txt', "w") as file:
        file.write("\n".join(val_annotations))
