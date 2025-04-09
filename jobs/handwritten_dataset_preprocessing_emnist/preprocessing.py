import cv2
from tqdm import tqdm
from emnist import extract_training_samples, extract_test_samples
import cv2
import numpy as np
from pathlib import Path
from ..logging_config import logger


def add_alpha(image: np.array):
    # Create a new RGBA image
    rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba[:,:,0] = image         # R channel (grayscale value)
    rgba[:,:,1] = image         # G channel (grayscale value)
    rgba[:,:,2] = image         # B channel (grayscale value)
    rgba[:,:,3] = 255 - image   # A channel (inverse of grayscale - white becomes transparent)

    return rgba


def crop_to_content(image):
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    content = np.argwhere(thresh == 255)
    x_min, x_max = np.min(content[:, 0]), np.max(content[:, 0])
    y_min, y_max = np.min(content[:, 1]), np.max(content[:, 1])

    return image[x_min:x_max, y_min:y_max]


def run_hw_preprocessing_emnist(dataset_type: str = "train"):
    if dataset_type == "train":
        letters_imgs, letters_labels = extract_training_samples("byclass")
    elif dataset_type == "test":
        letters_imgs, letters_labels = extract_test_samples("byclass")
    else:
        raise ValueError("dataset_type must be either 'train' or 'test'")
    
    logger.info(f"start preprocessing of {len(letters_imgs)} images for {dataset_type} dataset")

    hw_dataset_path = Path(f"./data/handwritten_dataset_emnist/{dataset_type}")
    counter = {}

    for image, label in tqdm(zip(letters_imgs, letters_labels), total=len(letters_imgs)):
        # check label is a number or uppercase letter
        if label in range(0, 10) or label in range(10, 36):
            counter[label] = counter.get(label, 0) + 1

            image = crop_to_content(image)

            # invert colors
            image = 255 - image

            image = add_alpha(image)

            # save image
            folder_name = chr(label+55) if label >= 10 else str(label)
            output_path = hw_dataset_path / folder_name
            output_path.mkdir(parents=True, exist_ok=True)
            res = cv2.imwrite(output_path / f"{counter[label]}.png", image)

            if not res:
                logger.error(f"Error saving image {counter[label]} in {folder_name}")
                continue
    
    logger.info(f"preprocessing complete")