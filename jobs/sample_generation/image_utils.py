import numpy as np
import cv2
import pandas as pd


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


def get_rand_string_image(rand_string):
    images_to_concat = []
    handwritten_dataset_base_path = "./data/handwritten_dataset"
    handwritten_dataset_processed_base_path = "./data/handwritten_dataset_processed"

    labels = pd.read_csv(f"{handwritten_dataset_base_path}/labels.csv")

    # select random images from handwritten dataset for each letter
    for character in rand_string:
        character_image_path = labels[labels["label"] == character].sample(n=1)["image"].iloc[0]
    
        character_image = cv2.imread(f"{handwritten_dataset_processed_base_path}/{character_image_path}", flags=cv2.IMREAD_UNCHANGED)
        images_to_concat.append(character_image)

    # pad images to fit the largest one
    max_height = max(map(lambda x: x.shape[0], images_to_concat))
    images_to_concat_padded = []
    
    for image in images_to_concat:
        pad_height = max_height - image.shape[0]
        
        padded_image = cv2.copyMakeBorder(
            image,
            top=pad_height, bottom=0, left=0, right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0, 0]  # Transparent (RGBA)
        )

        images_to_concat_padded.append(padded_image)

    return cv2.hconcat(images_to_concat_padded)