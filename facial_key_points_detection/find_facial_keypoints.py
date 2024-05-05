import numpy as np
import keras


def find_facial_key_points(image: np.ndarray) -> np.ndarray[float]:
    """
    Returns normalized facial kye points coordinates
    :param image: ndarray, three-dimensional array from image (image is black and white)
    :return: ndarray, two-dimensional array of predicted facial key points
    """
    model = keras.models.load_model('', {'mae': keras.losses.mean_absolute_error, 'huber': keras.losses.huber, 'log_cosh': keras.losses.log_cosh})
    result = model.predict(image)
    return result / 96
