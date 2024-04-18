import pandas as pd
import numpy as np


def convertor(img: np.array, img_size) -> np.array:
    img = np.reshape(img, (img_size, img_size, 1))
    img /= 255
    return img


def preprocessing_X(train_X: pd.DataFrame, img_size: int) -> np.array:
    train_X.ffill(inplace=True)
    m, n = train_X.shape
    img_lst = []
    for i in range(m):
        img = np.array(train_X['Image'][i].split(), dtype='float64')
        img = convertor(img, img_size)
        img_lst.append(img)
    img_array = np.array(img_lst)
    return img_array


def preprocessing_y(train_y: pd.DataFrame) -> np.array:
    keypoints_lst = []
    for i in range(len(train_y)):
        y = train_y.iloc[i, :].values
        keypoints_lst.append(y)
    keypoints_array = np.array(keypoints_lst, dtype='float')
    return keypoints_array
