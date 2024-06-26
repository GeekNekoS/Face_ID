from image.single_image_preprocessing import preprocess_image
import cv2
import numpy as np


def convert_image_small(image: np.ndarray, new_size: tuple[int, int] = (96, 96)) -> np.ndarray:
    """
    Converts image for small facial key points detection model.
    :param image: ndarray, array from image
    :param new_size: tuple[int, int], new image size [optional]
    :return: ndarray, converted image
    """
    converted_image = preprocess_image(image, new_size, cv2.COLOR_BGR2GRAY)
    converted_image = np.expand_dims(converted_image, axis=-1)
    return converted_image


def convert_image_medium(image: np.ndarray, new_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """
        Converts image for small facial key points detection model.
        :param image: ndarray, array from image
        :param new_size: tuple[int, int], new image size [optional]
        :return: ndarray, converted image
        """
    converted_image = preprocess_image(image, new_size, cv2.COLOR_BGR2GRAY)
    converted_image = np.expand_dims(converted_image, axis=-1)
    return converted_image