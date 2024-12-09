import cv2
import numpy as np
import string
import random
import json
from pathlib import Path
from tqdm import tqdm

from .api_utils import get_image_and_boxes, denormalise_box_coordinates
from .image_utils import get_rand_string_image, overlay_image


def generate_random_string(length, is_alphabetic, is_numeric):
    characters = ""
    
    if is_alphabetic:
        characters += string.ascii_uppercase
    
    if is_numeric:
        characters += string.digits
    
    return "".join(random.choices(characters, k=length))


def run_sampling():
    # TODO pass as job input
    document_id = "1"
    sample_size = 10
    sample_folder = Path(f"./data/sampling/document_{document_id}")

    sample_folder.mkdir(parents=True, exist_ok=True)

    document_image_buffer, boxes = get_image_and_boxes(document_id=document_id)
    document_image = cv2.imdecode(np.frombuffer(document_image_buffer, dtype=np.uint8), 
                                  flags=cv2.IMREAD_UNCHANGED)
    doc_height = document_image.shape[0]
    doc_width = document_image.shape[1]

    results_dict = {}

    for sample_id in tqdm(range(sample_size)):
        document_with_strings = document_image.copy()

        boxes_labels = []

        for box in boxes:
            string_length = int(np.random.normal(box["mean_length"], 1))
            rand_string = generate_random_string(string_length, box["is_alphabetic"], box["is_numeric"])
            boxes_labels.append({"box_id": box["id"],
                                 "box_name": box["name"],
                                 "generated_string": rand_string})

            rand_string_image = get_rand_string_image(rand_string)

            start_x, start_y, end_x, end_y = denormalise_box_coordinates(box["start_x_norm"], box["start_y_norm"], 
                                                                        box["end_x_norm"], box["end_y_norm"],
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
            new_width = int(string_width * scale)
            new_height = int(string_height * scale)
            
            # Resize the image
            rand_string_image_scaled = cv2.resize(rand_string_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            document_with_strings = overlay_image(background=document_with_strings, overlay=rand_string_image_scaled, position=(start_x, start_y))

        if cv2.imwrite(f"{sample_folder}/sample_{sample_id}.png", document_with_strings):
            pass
        else:
            raise ValueError("unable to save sample image")
        
        results_dict["sample_id"] = sample_id
        results_dict["boxes_labels"] = boxes_labels

    with open(f"{sample_folder}/sample_labels.json", "w") as json_file:
        json.dump(results_dict, json_file)