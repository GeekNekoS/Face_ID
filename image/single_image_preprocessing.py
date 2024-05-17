import cv2
import numpy as np


def resize_image(image: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """
    Resize image.
    :param image: ndarray, array from image
    :param new_size: tuple, new image size
    :return: ndarray, image with new size
    """
    image = cv2.resize(image, new_size)
    return image


def preprocess_image(image: np.ndarray, new_size: tuple[int, int], convert_color_options: int = None) -> np.ndarray:
    """
    Converts image for neural networks input.
    :param image: ndarray, array from image
    :param new_size: tuple, new image size
    :param convert_color_options: int, color converts option [optional]
    :return: ndarray, array from converted image
    """
    resized_image = resize_image(image, new_size)
    if convert_color_options is not None:
        resized_image = cv2.cvtColor(resized_image, convert_color_options)
    return resized_image
