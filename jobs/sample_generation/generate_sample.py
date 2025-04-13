import cv2
import numpy as np
import string
import random
from pathlib import Path
from tqdm import tqdm
import json

from .api_utils import get_image_and_boxes, publish_box_labels, publish_sample_image
from .image_utils import get_rand_string_image_emnist, add_string_image_to_document, normalise_box_coordinates
from .image_degradations import DegradationsConfig, ImageDegradator, GaussianBlurDegradation, \
    MotionBlurDegradation, GaussianNoiseDegradation, SaltPepperNoiseDegradation, BrightnessContrastDegradation, \
    WaveDistortionDegradation, ShadowDegradation, ColorFilterDegradation
from .letters_degradations import apply_gaussian_grayscale
from ..logging_config import logger


def generate_random_string(length, is_alphabetic, is_numeric):
    characters = ""
    
    if is_alphabetic:
        characters += string.ascii_uppercase
    
    if is_numeric:
        characters += string.digits
    
    return "".join(random.choices(characters, k=length))


def get_annotations(sample_labels: list, filename: str):
    boxes_annotations = map(lambda x: {"transcription": x["label"], "points": x["coords"]}, sample_labels)
    annotations = filename + "\t" + json.dumps(list(boxes_annotations))
    return annotations


def run_sampling(document_id: int, num_samples: int, version: str, publish: bool = False, dataset_type: str = "train"):
    logger.info(f"Generating {num_samples} for document_id: {document_id}, veriosn: {version}, dataset_type: {dataset_type}")
    logger.info(f"Publish to the server: {publish}")
    sample_folder = Path(f"./data/sampling/document_{document_id}/{version}/{dataset_type}/")
    label_folder = Path(f"./data/labels/document_{document_id}/{version}/{dataset_type}/")

    sample_folder.mkdir(parents=True, exist_ok=True)
    label_folder.mkdir(parents=True, exist_ok=True)

    document_image_buffer, boxes = get_image_and_boxes(document_id=document_id)
    document_image = cv2.imdecode(np.frombuffer(document_image_buffer, dtype=np.uint8), 
                                         flags=cv2.IMREAD_UNCHANGED)

    # Generate samples
    labels = {}
    degradations_list = [GaussianBlurDegradation, MotionBlurDegradation, GaussianNoiseDegradation, SaltPepperNoiseDegradation, 
                         BrightnessContrastDegradation, WaveDistortionDegradation, ShadowDegradation, ColorFilterDegradation]
    
    degradations_config = DegradationsConfig(config_path="./jobs/sample_generation/config/degradation_config.yaml")
    for sample_id in tqdm(range(num_samples)):
        document_with_strings = document_image.copy()
        blank_with_strings = np.full_like(document_with_strings, 255)
        # Generate random strings for each box and overlay them on the document
        boxes_labels = []
        for box in boxes:
            string_length = int(max(np.random.normal(box.mean_length, 1), 1))
            logger.debug(f"Box Name: {box.name}")
            logger.debug(f"random string parameters: length={string_length}, is_alphabetic={box.is_alphabetic}, is_numeric={box.is_numeric}")
            rand_string = generate_random_string(string_length, box.is_alphabetic, box.is_numeric)
            logger.debug(f"Generated random string: {rand_string}")

            rand_string_image = get_rand_string_image_emnist(rand_string, dataset_type)

            # Apply Degradations to the text image
            #erode_iterations = random.choice(np.linspace(10, 12, 3))
            #logger.debug(f"Applying erosion with {erode_iterations} iterations")
            #rand_string_image = erode_letter_image(rand_string_image, kernel_size=3, iterations=int(erode_iterations))

            # Apply Gaussian Grayscale to the string image
            if random.choice([True, False]):
                grayscale_mean = random.choice(np.linspace(0, 100, 101))
                logger.debug(f"Applying grayscale with {grayscale_mean} mean")
                rand_string_image = apply_gaussian_grayscale(rand_string_image, mean=int(grayscale_mean), std_dev=10)

            document_with_strings, blank_with_strings, box_label = add_string_image_to_document(document_image=document_with_strings,
                                                                                                blank_document=blank_with_strings,
                                                                                                box=box, 
                                                                                                rand_string=rand_string,
                                                                                                rand_string_image=rand_string_image)
            boxes_labels.append(box_label)
        # Apply Degradations to the document image
        n_degradations = random.randint(0, len(degradations_list))
        random_degradations = random.sample(degradations_list, n_degradations)

        image_degradator = ImageDegradator(image=document_with_strings, boxes_labels=boxes_labels, 
                                           degradations_list=random_degradations, degradations_config=degradations_config)

        degradated_image, boxes_labels = image_degradator.apply_degradations()
        
        # Save the sample image and labels
        sample_filename = f"sample_{sample_id}.png"
        logger.debug(f"Saving sample image: {sample_filename}")
        image_path= sample_folder / sample_filename
        if cv2.imwrite(image_path, degradated_image):
            pass
        else:
            raise ValueError("unable to save sample image")
        
        label_path = label_folder / sample_filename
        if cv2.imwrite(label_path, blank_with_strings):
            pass
        else:  
            raise ValueError("unable to save text only image")
        
        labels.update({sample_filename: boxes_labels})
        
        # Save Annotations in PaddleOCR format, TODO PaddleOCR should be responsible for this, converting from json to paddleocr format
        annotations = get_annotations(boxes_labels, sample_filename)
        with open(sample_folder.parent / f"{dataset_type}_labels.txt", "a") as file:
            file.write(annotations + "\n")
        
        # publish to the server
        if publish:
            logger.debug(f"Publishing sample {sample_id} image and boxes to the server")
            doc_height = document_image.shape[0]
            doc_width = document_image.shape[1]
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

    with open(sample_folder.parent / f"{dataset_type}_labels.json", "a") as labels_file:
        # Save the boxes labels in JSON format
        logger.debug(f"Saving sample labels: {sample_filename}")
        json.dump(labels, labels_file)