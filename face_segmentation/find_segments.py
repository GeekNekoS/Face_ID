import numpy as np
from face_segmentation.import_U_net import import_model
from face_segmentation.load_U_net import load_model
from face_segmentation.utils.settings import U_NET_CUSTOM_OBJECTS


def find_segments(image: np.ndarray, model_path=None, model=None, custom_objects: dict = U_NET_CUSTOM_OBJECTS) -> np.ndarray:
    """
    Find 11 face segments (background, skin, left eyebrow, right eyebrow, left eye, right eye, nose, upper lip, inner mouth, lower lip, hair).
    :param image: ndarray, array from image
    :param model_path: str, U-net model path [optional], if None, then the model will be downloaded from Yandex disk
    :param model: model, model for face segmentation [optional], if None, then the model will be downloaded from Yandex disk
    :param custom_objects: dict, custom objects for load models [optional], initially contains the necessary objects for the neural network from Yandex disk
    :return: ndarray, array from segments
    """
    if model_path is None and model is None:
        model = import_model(custom_objects)
    else:
        if model_path is not None:
            model = load_model(model_path, custom_objects)
    results = model.predict(image)
    return results