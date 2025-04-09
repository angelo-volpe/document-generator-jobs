import numpy as np
import cv2
import pandas as pd
import random
import os
from pathlib import Path

from ..logging_config import logger
from .models import Box


def overlay_image(background, overlay, position):
    """
    Overlay one image on top of another at a specific position.

    Args:
        background: Background image (BGR or BGRA format).
        overlay: Overlay image (BGR or BGRA format).
        position: Tuple (x, y) for the top-left corner where the overlay will be placed.

    Returns:
        Resultant image with the overlay.
    """
    x, y = position
    h, w = overlay.shape[:2]

    # If overlay has alpha channel, handle transparency
    if overlay.shape[2] == 4:
        # Split the overlay into RGB and alpha channels
        overlay_rgb = overlay[:, :, :3]
        alpha = overlay[:, :, 3] / 255.0  # Normalize alpha to [0, 1]

        # Extract the region of interest from the background
        roi = background[y:y+h, x:x+w]

        # Blend the overlay with the region of interest
        blended = (1.0 - alpha[..., None]) * roi + alpha[..., None] * overlay_rgb

        # Replace the region in the background with the blended image
        background[y:y+h, x:x+w] = blended.astype(np.uint8)
    else:
        # If no alpha channel, directly replace the region
        background[y:y+h, x:x+w] = overlay

    
    return background


def preprocess_images_to_concat(images_to_concat, processing: str):
    max_height = max(map(lambda x: x.shape[0], images_to_concat))
    images_to_concat_processed = []
    
    # pad images to fit the largest one
    if processing == "padding":
        for image in images_to_concat:
            pad_height = max_height - image.shape[0]
            
            padded_image = cv2.copyMakeBorder(
                image,
                top=pad_height, bottom=0, left=0, right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0, 0]  # Transparent (RGBA)
            )

            images_to_concat_processed.append(padded_image)
    
    # scale images to fit the largest one
    elif processing == "scaling":
        for image in images_to_concat:
            image_width = image.shape[1]
            image_height = image.shape[0]
            
            new_width = int(image_width * (max_height / image_height))
            new_height = max_height
            
            letter_scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            images_to_concat_processed.append(letter_scaled)

    else:
        raise ValueError("processing not valid")

    return images_to_concat_processed


def get_rand_string_image_emnist(rand_string: str, dataset_type: str):
    images_to_concat = []
    handwritten_dataset_base_path = Path(f"./data/handwritten_dataset_emnist/{dataset_type}")

    # select random images from handwritten dataset for each letter
    for character in rand_string:
        character_image_path = handwritten_dataset_base_path / character
        character_image_path = character_image_path / random.choice(os.listdir(character_image_path))
    
        character_image = cv2.imread(character_image_path, flags=cv2.IMREAD_UNCHANGED)

        # apply random small padding to the characters to make it more natural written
        if random.choice([True, False]):
            pad_size = int(random.choice(np.linspace(0, 10, 11)))
            pad = np.full((character_image.shape[0], pad_size, 4), np.array([255, 255, 255,   0]), dtype=np.uint8)
            character_image = cv2.hconcat([character_image, pad])

        images_to_concat.append(character_image)

    images_to_concat_processed = preprocess_images_to_concat(images_to_concat, processing="scaling")

    return cv2.hconcat(images_to_concat_processed)


def get_rand_string_image(rand_string):
    images_to_concat = []
    handwritten_dataset_base_path = "./data/handwritten_dataset"
    handwritten_dataset_processed_base_path = "./data/handwritten_dataset_processed"

    labels = pd.read_csv(f"{handwritten_dataset_base_path}/labels.csv")

    # select random images from handwritten dataset for each letter
    for character in rand_string:
        character_image_path = labels[labels["label"] == character].sample(n=1)["image"].iloc[0]
    
        character_image = cv2.imread(f"{handwritten_dataset_processed_base_path}/{character_image_path}", flags=cv2.IMREAD_UNCHANGED)

        # apply random small padding to the characters to make it more natural written
        if random.choice([True, False]):
            pad_size = int(random.choice(np.linspace(0, 70, 71)))
            pad = np.full((character_image.shape[0], pad_size, 4), np.array([255, 255, 255,   0]), dtype=np.uint8)
            character_image = cv2.hconcat([character_image, pad])

        images_to_concat.append(character_image)

    images_to_concat_processed = preprocess_images_to_concat(images_to_concat, processing="scaling")

    return cv2.hconcat(images_to_concat_processed)


def denormalise_box_coordinates(start_x_norm, start_y_norm, end_x_norm, end_y_norm, doc_width, doc_height):
    start_x = int(start_x_norm * doc_width)
    end_x = int(end_x_norm * doc_width)
    start_y = int(start_y_norm * doc_height)
    end_y = int(end_y_norm * doc_height)
    
    return start_x, start_y, end_x, end_y


def normalise_box_coordinates(start_x, start_y, end_x, end_y, doc_width, doc_height):
    start_x_norm = start_x / doc_width
    end_x_norm = end_x / doc_width
    start_y_norm = start_y / doc_height
    end_y_norm = end_y / doc_height
    
    return start_x_norm, start_y_norm, end_x_norm, end_y_norm


def add_string_image_to_document(document_image: np.array, 
                                 blank_document: np.array,
                                 box: Box, 
                                 rand_string: str, 
                                 rand_string_image: np.array):
    doc_height = document_image.shape[0]
    doc_width = document_image.shape[1]
    start_x, start_y, end_x, end_y = denormalise_box_coordinates(box.start_x_norm, box.start_y_norm, 
                                                                 box.end_x_norm, box.end_y_norm,
                                                                 doc_width=doc_width, 
                                                                 doc_height=doc_height)

    box_width = end_x - start_x
    box_height = end_y - start_y
    string_height, string_width = rand_string_image.shape[:2]
    
    # Calculate the scaling factor
    scale_w = box_width / string_width
    scale_h = box_height / string_height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    random_scale = random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    logger.debug(f"Applying random_scale: {random_scale}")
    new_width = int(string_width * scale * random_scale)
    new_height = int(string_height * scale * random_scale)
    
    # Resize the image to fit the box
    rand_string_image_scaled = cv2.resize(rand_string_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    pos_y_min = start_y
    pos_y_max = end_y-new_height
    pos_x_min = start_x
    pos_x_max = end_x-new_width
    new_start_x = int(random.choice(np.linspace(pos_x_min, pos_x_max, 10)))
    new_start_y = int(random.choice(np.linspace(pos_y_min, pos_y_max, 10)))
    document_with_strings = overlay_image(background=document_image, overlay=rand_string_image_scaled, 
                                            position=(new_start_x, new_start_y))
    
    blank_document_with_strings = overlay_image(background=blank_document, overlay=rand_string_image_scaled, 
                                                position=(new_start_x, new_start_y))

    new_end_x = new_start_x + new_width
    new_end_y = new_start_y + new_height

    # add buffer to boxes
    buffer = 1
    start_x_buffer, start_y_buffer = new_start_x-buffer, new_start_y-buffer
    end_x_buffer, end_y_buffer = new_end_x+buffer, new_end_y+buffer

    box_label = {"template_box_id": box.id,
                 "name": box.name,
                 "label": rand_string,
                 "coords": [[int(start_x_buffer), int(start_y_buffer)], [int(end_x_buffer), int(start_y_buffer)], 
                            [int(end_x_buffer), int(end_y_buffer)], [int(start_x_buffer), int(end_y_buffer)]]}

    return document_with_strings, blank_document_with_strings, box_label
