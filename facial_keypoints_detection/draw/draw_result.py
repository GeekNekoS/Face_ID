import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


def draw(img, result):
    img = np.reshape(img, (96, 96))
    plt.imshow(img[:, :], cmap='gray')
    for j in range(0, len(result), 2):
        plt.scatter(result[j], result[j+1], s=5, c='red', marker='o')
    plt.show()