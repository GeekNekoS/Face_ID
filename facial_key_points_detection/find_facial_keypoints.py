import numpy as np
from import_CNN import import_CNN
from dotenv import load_dotenv
load_dotenv('../yandex.env')


def find_facial_key_points(image: np.ndarray[int]) -> np.ndarray[float]:
    """
    Returns normalized facial kye points coordinates
    :param image: ndarray, three-dimensional array from image (image is black and white)
    :return: ndarray, two-dimensional array of predicted facial key points
    """
    model = import_CNN()
    result = model.predict(image)
    return result / 96
