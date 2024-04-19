import pandas as pd
import numpy as np


def convertor(img: np.array, img_size) -> np.array:
    img = np.reshape(img, (1, img_size, img_size))
    img /= 255
    return img


def preprocessing_X(X: pd.DataFrame, img_size: int) -> np.array:
    X.ffill(inplace=True)
    m, n = X.shape
    img_lst = []
    for i in range(m):
        img = np.array(X['Image'][i].split(), dtype='float32')
        img = convertor(img, img_size)
        img_lst.append(img)
    img_array = np.array(img_lst, dtype='float32')
    return img_array


def preprocessing_y(y: pd.DataFrame) -> np.array:
    keypoints_lst = []
    for i in range(len(y)):
        preprocessed_y = y.iloc[i, :].values
        keypoints_lst.append(preprocessed_y)
    keypoints_array = np.array(keypoints_lst, dtype='float32')
    return keypoints_array
