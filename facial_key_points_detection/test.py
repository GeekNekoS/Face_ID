from find_facial_key_points import find_facial_key_points
import cv2
from utils.single_image_preprocessing import convert_image
import matplotlib.pyplot as plt


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


dots = jpg_test('../test_data/test.jpg')
print(dots)
image = plt.imread('../test_data/test.jpg')
plt.imshow(image)
for i in range(0, len(dots[0]), 2):
    plt.scatter(dots[0][i], dots[0][i+1])
plt.show()
