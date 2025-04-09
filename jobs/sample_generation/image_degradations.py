import cv2
from PIL import Image, ImageFilter
import numpy as np
import random

from ..logging_config import logger


def gaussian_blur(image, kernel_size=1):
    gaussian_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return gaussian_blur


def motion_blur(image, kernel_size=5):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    motion_blur = cv2.filter2D(image, -1, kernel)

    return motion_blur


def gaussian_noise(image, mean=0, std=1):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def salt_pepper_noise(image, prob=0.01):
    noisy_image = image.copy()
    num_salt = np.ceil(prob * image.size * 0.5).astype(int)
    num_pepper = np.ceil(prob * image.size * 0.5).astype(int)
    
    # Add salt (white pixels)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # Add pepper (black pixels)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image


def brightness_contrast(image, alpha=1.2, beta=10):
    bright_contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return bright_contrast


# TODO let decide the user the orientation X or Y
def wave_distortion(image, boxes_labels, amplitude, frequency):
    rows, cols = image.shape[:2]

    Y, X = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    X_distorted = X + amplitude * np.sin(2 * np.pi * frequency * Y)
    Y_distorted = Y.copy()

    # Clip the coordinates to the image bounds
    X_distorted = np.clip(X_distorted, 0, cols - 1).astype(np.float32)
    Y_distorted = Y_distorted.astype(np.float32)

    # calculate new box areas
    for box in boxes_labels:
        coords = box["coords"]
        start_x = coords[0][0]
        end_x = coords[1][0]
        start_y = coords[0][1]
        end_y = coords[-1][1]
        
        # find where will be located the original x coordinates in the distorted image
        new_start_x = np.where(np.ceil(X_distorted[start_y:end_y,:]) == start_x)[1].min()
        new_end_x = np.where(np.ceil(X_distorted[start_y:end_y,:]) == end_x)[1].max()

        box["coords"] = [[int(new_start_x), int(start_y)], [int(new_end_x), int(start_y)], 
                         [int(new_end_x), int(end_y)], [int(new_start_x), int(end_y)]]
        
    distorted_image = cv2.remap(image, X_distorted, Y_distorted, interpolation=cv2.INTER_LINEAR)

    return distorted_image


def add_shadow(img: np.array, 
               direction: str ='diagonal', 
               intensity: float = 0.5, 
               blur: int = 100, 
               len_percentage: float = 0.5):
    """
    Add a natural-looking shadow to an image.
    
    Parameters:
    - img: numpy array of the image
    - direction: Shadow direction ('random', 'left', 'right', 'top', 'bottom')
    - intensity: Shadow intensity (0.0 to 1.0)
    - blur: Shadow blur amount
    - len_percentage
    """
    # Open the image
    
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    width, height = img.size
    
    # Create a shadow mask (initially same as original image)
    if img.mode == 'RGBA':
        # Use alpha channel if available
        r, g, b, a = img.split()
        shadow_mask = a.copy()
    else:
        # Create mask from image brightness
        gray = img.convert('L')
        shadow_mask = gray.point(lambda x: 255 if x > 10 else 0)

    # set the offset based on direction
    if direction == 'left':
        offset = (int(-width * len_percentage), 0)
    elif direction == 'right':
        offset = (int(width * len_percentage), 0)
    elif direction == 'top':
        offset = (0, int(-height * len_percentage))
    elif direction == 'bottom':
        offset = (0, int(height * len_percentage))
    elif direction == 'diagonal':
        offset_x = random.randint(int(-width * len_percentage), int(width * len_percentage))
        offset_y = random.randint(0, int(height * len_percentage))  # Shadows usually fall down
        offset = (offset_x, offset_y)
    else:
        raise ValueError('incompatible direction')
    
    # Create a new image for the shadow
    shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
    shadow_data = []
    
    # Create shadow by offsetting the mask
    shadow_img = Image.new('L', img.size, 0)
    shadow_img.paste(shadow_mask, offset)
    
    # Blur the shadow
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(int(blur)))
    
    # Adjust shadow intensity
    shadow_data = shadow_img.getdata()
    new_data = []
    for value in shadow_data:
        new_value = int(value * intensity)
        new_data.append(new_value)
    
    shadow_img.putdata(new_data)
    
    # Create final shadow image
    shadow.putalpha(shadow_img)
    
    result = Image.alpha_composite(img, shadow)
    result = np.array(result.convert("RGB"))
    
    return result

def apply_filter(p, factor):
        return min(255, int(factor * p))

apply_filter_vec = np.vectorize(apply_filter)

def apply_color_filter(img: np.array, red: float = 1.0, green: float = 1.0, blue: float = 1.0):
    """
    Apply a color filter to an image using PIL by adjusting RGB channels.
    
    Parameters:
    - img: np.array of the image to trasform
    - red: Multiplier for the red channel (1.0 is neutral)
    - green: Multiplier for the green channel (1.0 is neutral)
    - blue: Multiplier for the blue channel (1.0 is neutral)
    
    Returns:
    - np.array of the filtered image
    """
    for i, color in enumerate([red, green, blue]):
        if color != 1:
            img[:, :, i] = apply_filter_vec(img[:, :, i], color)

    return img


def apply_degradations(image: np.array, boxes_labels: list):
    # add degradations to the generated image
    degradations = [{"function": gaussian_blur, "parameters": {"kernel_size": np.arange(1, 5, 2)}}, 
                    {"function": motion_blur, "parameters": {"kernel_size": np.arange(1, 4, 1)}},
                    {"function": gaussian_noise, "parameters": {"std": np.arange(0.5, 1, 0.1)}},
                    {"function": salt_pepper_noise, "parameters": {"prob": np.arange(0.001, 0.01, 0.001)}},
                    {"function": brightness_contrast, "parameters": {"alpha": np.arange(0.8, 1.2, 0.1), 
                                                                        "beta": np.arange(-20, 20, 5)}},
                    {"function": wave_distortion, "parameters": {"amplitude": np.arange(1, 5, 1),
                                                                    "frequency": np.arange(0.001, 0.009, 0.001)}},
                    {"function": add_shadow, "parameters": {"direction": ["diagonal", "left", "right", "top", "bottom"],
                                                            "intensity": np.arange(0.1, 0.6, 0.1),
                                                            "blur": np.arange(50, 300, 50),
                                                            "len_percentage": np.arange(0.1, 0.9, 0.1)}},
                    {"function": apply_color_filter, "parameters": {"red": np.arange(0.9, 1.1, 0.01),
                                                                    "green": np.arange(0.9, 1.1, 0.01),
                                                                    "blue": np.arange(0.9, 1.1, 0.01)}}
                    ]

    n = random.randint(0, len(degradations))
    random_degradations = random.sample(degradations, n)

    degradated_image = image

    for degradation in random_degradations:
        params = {k: random.choice(v) for k, v in degradation["parameters"].items()}

        logger.debug(f"Applying degradation: {degradation['function'].__name__}, with parameters: {params}")

        if degradation["function"].__name__ == "wave_distortion":
            degradated_image = degradation["function"](degradated_image, **params, 
                                                       boxes_labels=boxes_labels)
        else:
            degradated_image = degradation["function"](degradated_image, **params)

    return degradated_image