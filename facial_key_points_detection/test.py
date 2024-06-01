from find_facial_key_points import find_facial_key_points
import cv2
from utils.single_image_preprocessing import convert_image_small
import matplotlib.pyplot as plt
import numpy as np
import keras


def png_test(img_path: str):
    image = cv2.imread(img_path)
    image = convert_image_small(image)
    result = find_facial_key_points(image)
    return result


def jpg_test(img_path: str):
    image = cv2.imread(img_path)
    image = convert_image_small(image)
    result = find_facial_key_points(image)
    return result


model = keras.models.load_model('models/FacialKeyPointsSmall(0.01).keras', {'mae': keras.losses.mean_absolute_error})
fig, axs = plt.subplots(1, 2)
image = cv2.imread('../test_data/test.png')
image_for_model = cv2.resize(image, (96, 96))
image_for_model = cv2.cvtColor(image_for_model, cv2.COLOR_BGR2GRAY)
image_for_model = np.expand_dims(image_for_model, 0)
image_for_model = np.expand_dims(image_for_model, -1)
dots2 = model.predict(image_for_model)
axs[0].imshow(image)
axs[1].imshow(image)
for j in range(0, len(dots2[0]) - 1, 2):
    axs[1].scatter(dots2[0][j] * image.shape[1], dots2[0][j + 1] * image.shape[0])
plt.show()
