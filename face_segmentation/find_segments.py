import numpy as np
from face_segmentation.import_U_net import import_model
from face_segmentation.load_U_net import load_model
from face_segmentation.utils.settings import U_NET_CUSTOM_OBJECTS


def find_segments(image: np.ndarray, model_path=None) -> np.ndarray:
    """
    Find 11 face segments (background, skin, left eyebrow, right eyebrow, left eye, right eye, nose, upper lip, inner mouth, lower lip, hair).
    :param image: ndarray, array from image
    :param model_path: str, U-net model path [optional], if None, then the model will be downloaded from Yandex disk
    :return: ndarray, array from segments
    """
    if model_path is None:
        model = import_model(U_NET_CUSTOM_OBJECTS)
    else:
        model = load_model(model_path, U_NET_CUSTOM_OBJECTS)
    results = model.predict(image)
    return results
