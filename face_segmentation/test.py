import cv2
from utils.single_image_preprocessing import convert_image
from find_segments import find_segments


def png_test(image_path):
    image = cv2.imread(image_path)
    converted_image = convert_image(image)
    result = find_segments(converted_image)
    return result


def jpg_test(image_path):
    image = cv2.imread(image_path)
    converted_image = convert_image(image)
    result = find_segments(converted_image)
    return result


png_test('../test_data/test.png')
jpg_test('../test_data/test.jpg')