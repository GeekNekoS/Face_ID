import numpy as np
from facial_key_points_detection.import_facial_key_points_model import import_model
from facial_key_points_detection.load_facial_key_points_model import load_model
from facial_key_points_detection.utils.settings import FACIAL_KEY_POINTS_MODEL_CUSTOM_OBJECTS


def find_facial_key_points(image: np.ndarray, model_path: str = None, custom_objects: dict = None) -> np.ndarray:
    """
    Returns normalized facial key points coordinates.
    :param image: ndarray, three-dimensional array from image (image is black and white)
    :param model_path: str, facial key points model path [optional], if None, then the model will be downloaded from Yandex disk
    :param custom_objects: dict, custom objects for load models [optional]
    :return: ndarray, two-dimensional array of predicted facial key points
    """
    if model_path is None:
        if custom_objects is None:
            model = import_model(FACIAL_KEY_POINTS_MODEL_CUSTOM_OBJECTS)
        else:
            model = import_model(custom_objects)
    else:
        if custom_objects is None:
            model = load_model(model_path, FACIAL_KEY_POINTS_MODEL_CUSTOM_OBJECTS)
        else:
            model = load_model(model_path, custom_objects)
    result = np.array(model.predict(image)[0])
    return result
