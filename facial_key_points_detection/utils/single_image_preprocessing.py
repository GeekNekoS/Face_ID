from image.single_image_preprocessing import preprocess_image
import cv2
import numpy as np


def convert_image(image: np.ndarray, new_size: tuple[int, int] = (96, 96)) -> np.ndarray:
    """
    Converts image for facial key points detection model.
    :param image: ndarray, array from image
    :param new_size: tuple[int, int], new image size [optional]
    :return: ndarray, converted image
    """
    converted_image = preprocess_image(image, new_size, cv2.COLOR_BGR2GRAY)
    return converted_image
