from image.single_image_preprocessing import preprocess_image
import cv2
import numpy as np
from PIL import Image


def convert_image(image: Image, new_size=(96, 96)) -> np.ndarray:
    converted_image = preprocess_image(image, new_size, cv2.COLOR_BGR2GRAY)
    return converted_image
