import numpy as np
from import_facial_key_points_model import import_CNN
import keras
custom_objects = {'mae': keras.losses.mean_absolute_error, 'huber': keras.losses.huber, 'log_cosh': keras.losses.log_cosh}


def find_facial_key_points(image: np.ndarray[int]) -> np.ndarray[float]:
    """
    Returns normalized facial kye points coordinates.
    :param image: ndarray, three-dimensional array from image (image is black and white)
    :return: ndarray, two-dimensional array of predicted facial key points
    """
    model = import_CNN(custom_objects)
    result = model.predict(image)
    return result / 96
