import cv2
import numpy as np
import math


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

        box["coords"] = [[new_start_x, start_y], [new_end_x, start_y], 
                         [new_end_x, end_y], [new_start_x, end_y]]
        
    distorted_image = cv2.remap(image, X_distorted, Y_distorted, interpolation=cv2.INTER_LINEAR)

    return distorted_image
