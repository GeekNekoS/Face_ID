from find_facial_key_points import find_facial_key_points
import cv2
from utils.single_image_preprocessing import convert_image


def png_test(img_path: str):
    image = cv2.imread(img_path)
    image = convert_image(image)
    result = find_facial_key_points(image)
    return result


def jpg_test(img_path: str):
    image = cv2.imread(img_path)
    image = convert_image(image)
    result = find_facial_key_points(image)
    return result


png_test('../test_data/test.png')
jpg_test('../test_data/test.jpg')
