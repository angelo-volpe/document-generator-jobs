import pandas as pd
import cv2
import logging
from tqdm import tqdm


def get_mask(image):
    # convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # invert gray image
    gray = 255 - gray

    return cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]


def add_alpha(image, mask):
    # make background transparent by placing the mask into the alpha channel
    new_img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    new_img[:, :, 3] = mask

    return new_img


def crop_to_content(image, mask):
    # get contours (presumably just one around the nonzero pixels) 
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cntr = contours[0]
    x,y,w,h = cv2.boundingRect(cntr)

    return image[y:y+h, x:x+w]


def run_hw_preprocessing():
    input_dataset_base_path = "./data/handwritten_dataset"
    output_dataset_base_path = "./data/handwritten_dataset_processed"
    labels = pd.read_csv(f"{input_dataset_base_path}/labels.csv")

    logging.info(f"start preprocessing of {len(labels)} images")

    for _, label in tqdm(labels.iterrows(), total=len(labels)):
        image_path = label["image"]
        image = cv2.imread(f'{input_dataset_base_path}/{image_path}')

        # process image
        mask = get_mask(image)
        image = add_alpha(image, mask)
        image = crop_to_content(image, mask)

        cv2.imwrite(f'{output_dataset_base_path}/{image_path}', image)
    
    logging.info(f"preprocessing complete")