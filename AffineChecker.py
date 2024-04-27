import keras
from facial_keypoints_detection.data_preprocessing.images_preprocessing import preprocessing
import pandas as pd
from face_alignment import ImageTransformation
import numpy as np


path = 'facial_keypoints_detection/data_for_learning/training.csv'
test_data = pd.read_csv(path)
model = keras.models.load_model('facial_keypoints_detection/CNNs/modelCNN(mae).h5', {'mae': keras.losses.mean_absolute_error, 'huber': keras.losses.huber, 'log_cosh': keras.losses.log_cosh})
X_train, y_train = preprocessing(test_data, 96)
results = model.predict(X_train)

for i in range(10):

    # path = f'{[i]}.jpg'
    # plt.imshow(X_train[i])
    # plt.savefig(path)
    # img = plt.imread(path)
    # print(img.shape)

    labels = (
    'left_eye_center_x','left_eye_center_y','right_eye_center_x','right_eye_center_y',
    'left_eye_inner_corner_x','left_eye_inner_corner_y','left_eye_outer_corner_x',
    'left_eye_outer_corner_y','right_eye_inner_corner_x','right_eye_inner_corner_y',
    'right_eye_outer_corner_x','right_eye_outer_corner_y','left_eyebrow_inner_end_x',
    'left_eyebrow_inner_end_y','left_eyebrow_outer_end_x','left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x','right_eyebrow_inner_end_y','right_eyebrow_outer_end_x',
    'right_eyebrow_outer_end_y','nose_tip_x','nose_tip_y','mouth_left_corner_x','mouth_left_corner_y',
    'mouth_right_corner_x','mouth_right_corner_y','mouth_center_top_lip_x','mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x','mouth_center_bottom_lip_y'
    )

    # my_json = ImageLabelsJSON(path, X_train[i], labels, path, tuple(results[i]))

    r_eye = [results[i][0], results[i][1]]
    nose = [results[i][20], results[i][21]]
    l_eye = [results[i][2], results[i][3]]

    face_parts = np.array([l_eye, nose, r_eye])

    base_face_parts_coordinates = np.array([[27.2, 35.0], [50.0, 55.0], [70.4, 35.0]]).astype(np.float32)

    transformation = ImageTransformation(np.array(X_train[i]))
    transformation.face_alignment(face_parts)

    transformation.draw_start_image()
    transformation.draw_transformed_image()

    # os.remove(path)

