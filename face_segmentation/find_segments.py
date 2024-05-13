import numpy as np
from import_U_net import import_model
from face_segmentation.utils.settings import U_NET_CUSTOM_OBJECTS


def find_segments(image: np.ndarray) -> np.ndarray:
    """
    Find 11 face segments (background, skin, left eyebrow, right eyebrow, left eye, right eye, nose, upper lip, inner mouth, lower lip, hair).
    :param image: ndarray, array from image
    :return: ndarray, array from segments
    """
    model = import_model(U_NET_CUSTOM_OBJECTS)
    results = model.predict(image)
    return results
