import numpy as np
import matplotlib.pyplot as plt


def draw(img: np.ndarray, result: np.ndarray, rows: int, columns: int) -> None:
    """
    Draw facial key points on black and white image.
    :param img: ndarray, one-dimensional ndarray from image
    :param result: ndarray, two-dimensional ndarray of facial key points
    :param rows: int, rows number in image
    :param columns: int, columns number in image
    :return: None
    """
    img = np.reshape(img, (rows, columns))
    plt.imshow(img, cmap='gray')
    for j in range(0, len(result), 2):
        plt.scatter(result[j], result[j+1], s=5, c='red', marker='o')
    plt.show()
