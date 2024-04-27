import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image


def get_3d_facial_keypoints_large(image_path):
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    img = mp.Image.create_from_file(image_path)
    results = detector.detect(img)
    return results.face_landmarks, results.face_blendshapes, results.facial_transformation_matrixes


def get_3d_facial_keypoints_light(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    while True:
        img = cv2.imread(image_path)
        results = face_mesh.process(img)
        return results.multi_face_landmarks


test_img = plt.imread('test_face.jpg')
base_landmarks, base_blendshapes, base_matrixes = get_3d_facial_keypoints_large('base_face.png')
test_landmarks, test_blenshape, test_matrixes = get_3d_facial_keypoints_large('test_face.jpg')
print(base_landmarks[0], base_blendshapes[0], base_matrixes[0], sep='\n_______\n')
base_indexes = [4, 23, 253]
base_coordinates = []
test_coordinates = []
all_coordinates = []
for i in range(len(base_landmarks[0])):
    all_coordinates.append([base_landmarks[0][i].x, base_landmarks[0][i].y, base_landmarks[0][i].z])
all_coordinates = np.float32(all_coordinates)
all_coordinates = all_coordinates.copy(order='C')
for i in base_indexes:
    base_coordinates.append([base_landmarks[0][i].x, base_landmarks[0][i].y, base_landmarks[0][i].z])
    test_coordinates.append([test_landmarks[0][i].x, base_landmarks[0][i].y, base_landmarks[0][i].z])
base_coordinates = np.float32(base_coordinates)
base_coordinates = base_coordinates.copy(order='C')
test_coordinates = np.float32(test_coordinates)
test_coordinates = test_coordinates.copy(order='C')
print(base_coordinates)
print('_'*35)
print(test_coordinates)
print('_'*35)
print(all_coordinates)
print(all_coordinates.shape)
print('_'*35)
rows, columns, = base_coordinates.shape
matrix = cv2.getAffineTransform(test_coordinates, base_coordinates)
print(matrix)
result = cv2.warpAffine(all_coordinates, matrix, (rows, columns))
print(result)
plt.imshow(test_img)
for i in range(len(result)):
    plt.scatter(result[i][0] * test_img.shape[1], result[i][1] * test_img.shape[0], c='red', marker='o')
plt.show()
