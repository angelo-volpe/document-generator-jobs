import cv2
import numpy as np
import string
import random
from pathlib import Path
from tqdm import tqdm

from .api_utils import get_image_and_boxes, publish_box_labels, publish_sample_image
from .image_utils import get_rand_string_image, overlay_image, denormalise_box_coordinates, normalise_box_coordinates
from .degradation import gaussian_blur, motion_blur, gaussian_noise, salt_pepper_noise, brightness_contrast, wave_distortion


def generate_random_string(length, is_alphabetic, is_numeric):
    characters = ""
    
    if is_alphabetic:
        characters += string.ascii_uppercase
    
    if is_numeric:
        characters += string.digits
    
    return "".join(random.choices(characters, k=length))


def run_sampling(document_id, num_samples):
    sample_folder = Path(f"./data/sampling/document_{document_id}")

    sample_folder.mkdir(parents=True, exist_ok=True)

    document_image_buffer, boxes = get_image_and_boxes(document_id=document_id)
    document_image = cv2.imdecode(np.frombuffer(document_image_buffer, dtype=np.uint8), 
                                  flags=cv2.IMREAD_UNCHANGED)
    doc_height = document_image.shape[0]
    doc_width = document_image.shape[1]

    for sample_id in tqdm(range(num_samples)):
        document_with_strings = document_image.copy()

        boxes_labels = []
        for box in boxes:
            string_length = int(np.random.normal(box["mean_length"], 1))
            rand_string = generate_random_string(string_length, box["is_alphabetic"], box["is_numeric"])

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
            random_scale = random.choice([0.7, 0.8, 0.9, 1])
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
            document_with_strings = overlay_image(background=document_with_strings, overlay=rand_string_image_scaled, 
                                                  position=(new_start_x, new_start_y))

            new_end_x = new_start_x + new_width
            new_end_y = new_start_y + new_height

            # add buffer to boxes
            buffer = 1
            start_x_buffer, start_y_buffer = new_start_x-buffer, new_start_y-buffer
            end_x_buffer, end_y_buffer = new_end_x+buffer, new_end_y+buffer

            boxes_labels.append({"template_box_id": box["id"],
                                 "name": box["name"],
                                 "label": rand_string,
                                 "coords": [[start_x_buffer, start_y_buffer], [end_x_buffer, start_y_buffer], 
                                            [end_x_buffer, end_y_buffer], [start_x_buffer, end_y_buffer]]
                                 })
                    
        # add degradations to the generated image
        degradations = [{"function": gaussian_blur, "parameters": {"kernel_size": np.arange(1, 5, 2)}}, 
                {"function": motion_blur, "parameters": {"kernel_size": np.arange(1, 4, 1)}},
                {"function": gaussian_noise, "parameters": {"std": np.arange(0.5, 1, 0.1)}},
                {"function": salt_pepper_noise, "parameters": {"prob": np.arange(0.001, 0.01, 0.001)}},
                {"function": brightness_contrast, "parameters": {"alpha": np.arange(0.8, 1.2, 0.1), 
                                                                 "beta": np.arange(-20, 20, 5)}},
                {"function": wave_distortion, "parameters": {"amplitude": np.arange(1, 5, 1),
                                                             "frequency": np.arange(0.001, 0.009, 0.001)}},
                 ]

        n = random.randint(0, len(degradations))
        random_degradations = random.sample(degradations, n)

        degradated_image = document_with_strings

        for degradation in random_degradations:
            params = {k: random.choice(v) for k, v in degradation["parameters"].items()}

            if degradation["function"].__name__ == "wave_distortion":
                degradated_image = degradation["function"](degradated_image, **params, 
                                                           boxes_labels=boxes_labels)
            else:
                degradated_image = degradation["function"](degradated_image, **params)

        image_path = f"{sample_folder}/sample_{sample_id}.png"

        # TODO use only for debugging
        if cv2.imwrite(image_path, degradated_image):
            pass
        else:
            raise ValueError("unable to save sample image")
        
        for box in boxes_labels:
            coords = box.pop("coords")
            
            start_x = coords[0][0]
            end_x = coords[1][0]
            start_y = coords[0][1]
            end_y = coords[-1][1]

            # normalise the box coordinates
            start_x_norm, start_y_norm, end_x_norm, end_y_norm = normalise_box_coordinates(start_x, start_y, end_x, end_y,
                                                                                           doc_width, doc_height)
            
            box["start_x_norm"] = start_x_norm
            box["start_y_norm"] = start_y_norm
            box["end_x_norm"] = end_x_norm
            box["end_y_norm"] = end_y_norm
        
        sample_document_id = publish_sample_image(image_path, sample_id, document_id)
        publish_box_labels(boxes_labels, sample_document_id)