import cv2
from PIL import Image, ImageFilter
import numpy as np
import random
from pathlib import Path
import yaml

from abc import ABC, abstractmethod
from typing import Dict, List


from jobs.logging_config import logger

class ImageDegradation(ABC):
    
    def __init__(self, parameters: Dict[str, any]):
        self.parameters = parameters

    def apply(self, image: np.array, boxes_labels: list = None):
        """
        Apply the degradation to the image.
        Parameters:
        - image: Input image as a numpy array
        - boxes_labels: List of dictionaries containing box coordinates and labels, 
                        only needed in case the degradation change the box coordinates
        """
        logger.debug(f"Applying degradation {self.__class__.__name__} with parameters: {self.parameters}")
        # save original image
        image = image.copy()
        if boxes_labels is not None:
            degradated_image, new_boxes_labels = self._apply_degradation(image, boxes_labels=boxes_labels, **self.parameters)
            return degradated_image, new_boxes_labels
        else:
            degradated_image = self._apply_degradation(image, **self.parameters)
            return degradated_image
    
    @abstractmethod
    def _apply_degradation(self, image: np.array, **kwargs):
        pass


class GaussianBlurDegradation(ImageDegradation):

    def _apply_degradation(self, image: np.array, kernel_size: int):
        gaussian_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return gaussian_blur
    

class MotionBlurDegradation(ImageDegradation):
    
    def _apply_degradation(self, image: np.array, kernel_size: int):
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        motion_blur = cv2.filter2D(image, -1, kernel)

        return motion_blur
    

class GaussianNoiseDegradation(ImageDegradation):

    def _apply_degradation(self, image: np.array, mean: float = 0, std: float = 1):
        noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image


class SaltPepperNoiseDegradation(ImageDegradation):

    def _apply_degradation(self, image: np.array, prob: float = 0.01):
        num_salt = np.ceil(prob * image.size * 0.5).astype(int)
        num_pepper = np.ceil(prob * image.size * 0.5).astype(int)
        
        # Add salt (white pixels)
        coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
        image[coords[0], coords[1]] = 255

        # Add pepper (black pixels)
        coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
        image[coords[0], coords[1]] = 0

        return image


class BrightnessContrastDegradation(ImageDegradation):

    def _apply_degradation(self, image: np.array, alpha: float = 1.2, beta: int = 10):
        bright_contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return bright_contrast


class WaveDistortionDegradation(ImageDegradation):
    
    def _apply_degradation(self, image: np.array, boxes_labels: list, amplitude: float, frequency: float):
        rows, cols = image.shape[:2]

        Y, X = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

        X_distorted = X + amplitude * np.sin(2 * np.pi * frequency * Y)
        Y_distorted = Y.copy()

        # Clip the coordinates to the image bounds
        X_distorted = np.clip(X_distorted, 0, cols - 1).astype(np.float32)
        Y_distorted = Y_distorted.astype(np.float32)

        new_boxes_labels = []
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

            new_box = box.copy()
            new_box["coords"] = [[int(new_start_x), int(start_y)], [int(new_end_x), int(start_y)], 
                                 [int(new_end_x), int(end_y)], [int(new_start_x), int(end_y)]]
            
            new_boxes_labels.append(new_box)
            
        distorted_image = cv2.remap(image, X_distorted, Y_distorted, interpolation=cv2.INTER_LINEAR)

        return distorted_image, new_boxes_labels
    

class ShadowDegradation(ImageDegradation):

    def _apply_degradation(self, image: np.array, direction: str ='diagonal', intensity: float = 0.5, blur: int = 100, len_percentage: float = 0.5):
        """
        Add a natural-looking shadow to an image.
        
        Parameters:
        - image: Input image as a numpy array
        - direction: Shadow direction ('random', 'left', 'right', 'top', 'bottom')
        - intensity: Shadow intensity (0.0 to 1.0)
        - blur: Shadow blur amount
        - len_percentage: Length of the shadow as a percentage of the image size (0.0 to 1.0)
        """
        # Open the image
        
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
        width, height = image.size
        
        # Create a shadow mask (initially same as original image)
        if image.mode == 'RGBA':
            # Use alpha channel if available
            r, g, b, a = image.split()
            shadow_mask = a.copy()
        else:
            # Create mask from image brightness
            gray = image.convert('L')
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
        shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
        shadow_data = []
        
        # Create shadow by offsetting the mask
        shadow_img = Image.new('L', image.size, 0)
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
        
        result = Image.alpha_composite(image, shadow)
        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        return result


class ColorFilterDegradation(ImageDegradation):

    def __init__(self, parameters: Dict[str, any]):
        super().__init__(parameters)

        def apply_filter(p, factor):
            return min(255, int(factor * p))
        
        self.apply_filter_vec = np.vectorize(apply_filter)
    
    def _apply_degradation(self, image: np.array, red: float = 1.0, green: float = 1.0, blue: float = 1.0):
        """
        Apply a color filter to an image using PIL by adjusting RGB channels.
        
        Parameters:
        - red: Multiplier for the red channel (1.0 is neutral)
        - green: Multiplier for the green channel (1.0 is neutral)
        - blue: Multiplier for the blue channel (1.0 is neutral)
        
        Returns:
        - np.array of the filtered image
        """
        for i, color in enumerate([red, green, blue]):
            if color != 1:
                image[:, :, i] = self.apply_filter_vec(image[:, :, i], color)

        return image


class DegradationsConfig():
    def __init__(self, config_path: Path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Load parameters from the config file
        self.degradation_params_range = {}
        for degradation, parameters in config.items():
            params_range = {}
            for param_name, param_value in parameters.items():
                if isinstance(param_value, dict):
                    # If the value is a dictionary, create a range
                    params_range[param_name] = np.arange(param_value["min"], param_value["max"], param_value["step"]).tolist()
                elif isinstance(param_value, list):
                    # If the value is a list, use it directly
                    params_range[param_name] = param_value
                elif isinstance(param_value, (int, float, str)):
                    # If the value is a single number, create a range with one value
                    params_range[param_name] = [param_value]
                else:
                    raise ValueError(f"Invalid configuration for {degradation}: {param_value}")
            self.degradation_params_range[degradation] = params_range

    def get_params_range(self, degradation: ImageDegradation) -> Dict[str, any]:
        degradation_name = degradation.__name__
        if degradation_name not in self.degradation_params_range:
            raise ValueError(f"Degradation {degradation_name} not found in config.")
        
        params_range = self.degradation_params_range[degradation_name]
        return params_range
    


class ImageDegradator:
    def __init__(self, image: np.array, boxes_labels: List[dict], degradations_list: List[ImageDegradation], degradations_config: DegradationsConfig):
        self.image = image
        self.degradations_list = degradations_list
        self.boxes_labels = boxes_labels
        self.degradations_config = degradations_config

    def apply_degradations(self):
        for degradation in self.degradations_list:
            params_range = self.degradations_config.get_params_range(degradation)

            params = {}
            for param, values_range in params_range.items():
                params[param] = random.choice(values_range)

            degradation_instance = degradation(parameters=params)

            logger.debug(f"Applying degradation: {degradation_instance.__class__.__name__}")
            
            # Apply the degradation
            if isinstance(degradation_instance, WaveDistortionDegradation):
                self.image, self.boxes_labels = degradation_instance.apply(image=self.image, boxes_labels=self.boxes_labels)
            else:
                self.image = degradation_instance.apply(self.image)
        return self.image, self.boxes_labels
