from image.single_image_preprocessing import preprocess_image
import cv2
import numpy as np
from PIL import Image


def convert_image(image: Image, new_size=(96, 96)) -> np.ndarray:
    converted_image = preprocess_image(image, new_size, cv2.COLOR_BGR2GRAY)
    np.expand_dims(converted_image, axis=-1)
    return converted_image
