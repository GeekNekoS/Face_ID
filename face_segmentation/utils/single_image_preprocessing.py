from image.single_image_preprocessing import preprocess_image
import cv2
import numpy as np


def convert_image(image, new_size=(512, 512)):
    """
    Converts image for face segmentation model.
    :param image: ndarray, array from image
    :param new_size: tuple[int, int], new image size [optional]
    :return: ndarray, converted image
    """
    converted_image = preprocess_image(image, new_size, cv2.COLOR_BGR2RGB)
    converted_image /= 255
    converted_image = np.expand_dims(converted_image, axis=0)
    return converted_image