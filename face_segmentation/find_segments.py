import numpy as np
from import_face_segmentation_model import import_U_net


def find_segments(image: np.ndarray) -> np.ndarray:
    """
    Find 11 face segments
    :param image: ndarray, array from image
    :return: ndarray, array from segments
    """
    model = import_U_net()
    results = model.predict(image)
    return results
