import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
from face_recognition.detect_face import detect_face


def scatter_creator(img_path: str):
    model_path = 'facial_key_points_detection/models/FacialKeyPointsSmall(0.04)(with bug dataset).keras'
    FACIAL_KEY_POINTS_MODEL_CUSTOM_OBJECTS = {'mae': keras.losses.mean_absolute_error}
    model = keras.models.load_model(model_path, FACIAL_KEY_POINTS_MODEL_CUSTOM_OBJECTS)

    img = detect_face(img_path, img_path)
    # img = plt.imread(img_path)
    img_shape = img.shape

    img1 = cv2.resize(img, (96, 96))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = np.expand_dims(img1, 0)

    # img2 = cv2.resize(img, (256, 256))
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # img2 = np.expand_dims(img2, 0)
    #
    # img3 = cv2.resize(img, (512, 512))
    # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    # img3 = np.expand_dims(img3, 0)


    coordinates1 = model.predict(img1)
    # print(coordinates1)
    # coordinates2 = model.predict(img2)
    # coordinates3 = model.predict(img3)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img)

    for i in range(0, len(coordinates1[0])-1, 2):
        coordinates1[0][i] = coordinates1[0][i] * img_shape[1]
        coordinates1[0][i+1] = coordinates1[0][i+1] * img_shape[0]
        # print(coordinates1[0][i], coordinates1[0][i+1])
        axs[0].scatter(coordinates1[0][i], coordinates1[0][i+1])

    # axs[1].imshow(img2)
    # for i in range(0, len(coordinates2)-1, 2):
    #     coordinates2[i] = coordinates2[i] * img_shape[0]
    #     coordinates2[i+1] = coordinates2[i+1] * img_shape[1]
    #     plt.scatter(coordinates2[i], coordinates2[i+1])

    # axs[1].imshow(img3)
    # for i in range(0, len(coordinates3) - 1, 2):
    #     coordinates3[i] = coordinates3[i] * img_shape[0]
    #     coordinates3[i+1] = coordinates3[i+1] * img_shape[1]
    #     plt.scatter(coordinates3[i], coordinates3[i + 1])
    plt.show()


if __name__ == '__main__':
    scatter_creator('images/examples/vlad_a4.jpg')
