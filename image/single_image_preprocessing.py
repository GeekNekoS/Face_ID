import cv2
import numpy as np


def resize_image(image, new_size: tuple[int, int]):
    """
    Resize image.
    :param image: PIL Image, image
    :param new_size: tuple, new image size
    :return: PIL Image, image with new size
    """
    image = cv2.resize(image, new_size)
    return image


def preprocess_image(image, new_size, convert_color_options=None):
    resized_image = resize_image(image, new_size)
    if convert_color_options is not None:
        resized_image = cv2.cvtColor(resized_image, convert_color_options)
    resized_image = np.expand_dims(resized_image, axis=0)
    return resized_image
