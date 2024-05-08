import numpy as np
import keras
from import_face_segmentation_model import import_U_net
custom_objects = {'categorical_crossentropy': keras.losses.categorical_crossentropy}


def find_segments(image: np.ndarray) -> np.ndarray:
    """
    Find 11 face segments.
    :param image: ndarray, array from image
    :return: ndarray, array from segments
    """
    model = import_U_net(custom_objects)
    results = model.predict(image)
    return results
