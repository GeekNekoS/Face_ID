import numpy as np
from import_facial_key_points_model import import_model
from facial_key_points_detection.utils.settings import FACIAL_KEY_POINTS_MODEL_CUSTOM_OBJECTS


def find_facial_key_points(image: np.ndarray) -> np.ndarray:
    """
    Returns normalized facial kye points coordinates.
    :param image: ndarray, three-dimensional array from image (image is black and white)
    :return: ndarray, two-dimensional array of predicted facial key points
    """
    model = import_model(FACIAL_KEY_POINTS_MODEL_CUSTOM_OBJECTS)
    result = model.predict(image)
    return result / 96
