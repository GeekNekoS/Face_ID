from find_facial_key_points import find_facial_key_points
import cv2
from utils.single_image_preprocessing import convert_image


def path_png_test(img_path: str):
    image = cv2.imread(img_path)
    image = convert_image(image)
    result = find_facial_key_points(image)
    return result


def path_jpg_test(img_path: str):
    image = cv2.imread(img_path)
    image = convert_image(image)
    result = find_facial_key_points(image)
    return result


path_png_test('test.png')
path_jpg_test('test.jpg')
