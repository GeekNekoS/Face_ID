import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from copy import deepcopy


def data_augmentation(img_array: np.array, keypoints_array: np.array, img_size: int) -> tuple[np.array, np.array]:
    flipped_img_array = img_array[:, :, :, ::-1]
    flipped_img_array = np.array(flipped_img_array)
    flipped_keypoints_array = deepcopy(keypoints_array)
    for i in range(0, keypoints_array.shape[1], 2):
        flipped_keypoints_array[:, i] = abs(img_size - flipped_keypoints_array[:, i])
    img_array = np.concatenate((img_array, flipped_img_array))
    keypoints_array = np.concatenate((keypoints_array, flipped_keypoints_array))
    return img_array, keypoints_array


def convertor(img: np.array, img_size) -> np.array:
    img = np.reshape(img, (1, img_size, img_size))
    img /= 255
    return img


def preprocessing(dataset: pd.DataFrame, img_size: int) -> tuple[np.array, np.array]:
    img_array = preprocessing_X(dataset, img_size)
    dataset.drop('Image', axis=1, inplace=True)
    keypoints_array = preprocessing_y(dataset)
    return img_array, keypoints_array


def preprocessing_X(X: pd.DataFrame, img_size: int) -> np.array:
    X.ffill(inplace=True)
    m = X.shape[0]
    img_lst = []
    for i in range(m):
        img = np.array(X['Image'][i].split(), dtype='float32')
        img = convertor(img, img_size)
        img_lst.append(img)
    img_array = np.array(img_lst, dtype='float32')

    print(img_array.shape)

    return img_array


def preprocessing_y(y: pd.DataFrame) -> np.array:
    keypoints_lst = []
    for i in range(len(y)):
        preprocessed_y = y.iloc[i, :].values
        keypoints_lst.append(preprocessed_y)
    keypoints_array = np.array(keypoints_lst, dtype='float32')
    return keypoints_array
