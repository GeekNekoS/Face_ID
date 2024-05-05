import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def convertor(img: np.ndarray, rows: int, columns: int) -> np.ndarray:
    """
    Converts a one-dimensional array to a two-dimensional array.
    :param img: ndarray, one-dimensional array from image
    :param rows: int, rows number in image
    :param columns: int, column number in image
    :return: ndarray, two-dimensional array
    """
    img = np.reshape(img, (rows, columns, 1))
    img /= 255
    return img


def preprocessing(dataset: pd.DataFrame, rows: int, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates ndarray of converted images from DataFrame of images and creates a ndarray from a DataFrame of facial key points.
    :param dataset: DataFrame, DataFrame of images, which has a column called "Images" containing images in the one-dimensional ndarray view. Which also had second column with facial key points coordinates.
    :param rows: int, rows number in image
    :param columns: int, column number in image
    :return: tuple[ndarray, ndarray], X and y data for NN train
    """
    img_array = preprocessing_X(dataset, rows, columns)
    dataset.drop('Image', axis=1, inplace=True)
    keypoints_array = preprocessing_y(dataset)
    return img_array, keypoints_array


def preprocessing_X(X: pd.DataFrame, rows: int, columns: int) -> np.ndarray:
    """
    Creates ndarray of converted images from DataFrame of images.
    :param X: DataFrame, DataFrame of images, which has a column called "Images" containing images in the one-dimensional ndarray view
    :param rows: int, rows number in image
    :param columns: int, column number in image
    :return: ndarray, three-dimensional array of images
    """
    X.ffill(inplace=True)
    m = X.shape[0]
    img_lst = []
    for i in range(m):
        img = np.array(X['Image'][i].split(), dtype='float32')
        img = convertor(img, rows, columns)
        img_lst.append(img)
    img_array = np.array(img_lst, dtype='float32')
    return img_array


def preprocessing_y(y: pd.DataFrame) -> np.ndarray:
    """
    Creates a ndarray from a DataFrame of facial key points.
    :param y: DataFrame, facial key points
    :return: ndarray, two-dimensional facial key points array
    """
    keypoints_lst = []
    for i in range(len(y)):
        preprocessed_y = y.iloc[i, :].values
        keypoints_lst.append(preprocessed_y)
    keypoints_array = np.array(keypoints_lst, dtype='float32')
    return keypoints_array


def make_pipeline(dataset: pd.DataFrame, rows: int, columns: int, batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Process input DataFrame in train data and validation data for train NN.
    :param dataset: DataFrame, DataFrame, DataFrame of images, which has a column called "Images" containing images in the one-dimensional ndarray view. Which also had second column with facial key points coordinates.
    :param rows: int, rows number in image
    :param columns: int, column number in image
    :param batch_size: int, batch size
    :return: tuple[Dataset, Dataset], zipped Datasets for train and validation
    """
    img_array, keypoints_array = preprocessing(dataset, rows, columns)
    X_train, X_val, y_train, y_val = train_test_split(img_array, keypoints_array, test_size=0.3)
    X_train = tf.data.Dataset.from_tensor_slices(X_train)
    X_val = tf.data.Dataset.from_tensor_slices(X_val)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    y_val = tf.data.Dataset.from_tensor_slices(y_val)
    zipped_train = tf.data.Dataset.zip((X_train, y_train)).batch(batch_size)
    zipped_test = tf.data.Dataset.zip((X_val, y_val)).batch(batch_size)
    return zipped_train, zipped_test