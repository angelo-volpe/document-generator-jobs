import cv2
import numpy as np


def erode_letter_image(img, kernel_size=3, iterations=1):
    """
    Apply erosion to an image and invert colors.
    
    Parameters:
    - image_path: Path to the input image
    - kernel_size: Size of the erosion kernel (default is 3x3)
    
    Returns:
    - Original image
    - Eroded image
    - Inverted eroded image
    """    
    # Create a binary threshold to ensure pure black and white
    # Note the THRESH_BINARY_INV to make letter white, background black
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Create a rectangular erosion kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply erosion
    eroded_img = cv2.erode(binary_img, kernel, iterations=iterations)
    
    # Invert the eroded image (now white background, black letter)
    inverted_eroded_img = cv2.bitwise_not(eroded_img)
    
    return inverted_eroded_img


def apply_gaussian_grayscale(img, mean=128, std_dev=30):
    """
    Apply a Gaussian distribution to letter grayscale intensity on a white background.
    
    Parameters:
    - img: image to apply the Gaussian distribution to
    - mean: Center of the Gaussian distribution (default 128 - middle gray)
    - std_dev: Standard deviation of the Gaussian distribution
    
    Returns:
    - Original binary image
    - Gaussian-distributed grayscale image
    """
    # Create a binary threshold to ensure white background, black letter
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Create a mask for the letter (zero pixels)
    letter_mask = binary_img == 0
    
    # Create white background image
    gaussian_img = np.full_like(img, 255, dtype=np.float32)
    
    # Apply Gaussian distribution to letter pixels
    if np.any(letter_mask):
        # Generate Gaussian noise
        gaussian_noise = np.random.normal(
            loc=mean, 
            scale=std_dev, 
            size=img.shape
        )
        
        # Apply the noise only to letter pixels
        gaussian_img[letter_mask] = gaussian_noise[letter_mask]
        
        # Clip values to valid grayscale range
        gaussian_img = np.clip(gaussian_img, 0, 255).astype(np.uint8)

    return gaussian_img