from image.resize import resize_image
import cv2
import numpy as np


def preprocess_image(image, new_size, convert_color_options=None):
    resized_image = resize_image(image, new_size)
    if convert_color_options is not None:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    resized_image = np.expand_dims(resized_image, axis=0)
    return resized_image