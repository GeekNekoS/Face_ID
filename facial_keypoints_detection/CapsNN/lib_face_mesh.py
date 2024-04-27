import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from sympy import *
from sympy.solvers.solveset import linsolve
from numpy.linalg import inv
from numpy import matmul
from numpy import transpose


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


def find_transition_matrix(start_points, end_points):
    v1, v2, v3 = start_points[0], start_points[1], start_points[2]
    v1k, v2k, v3k = end_points[0], end_points[1], end_points[2]
    x1, y1, z1 = v1[0], v1[1], v1[2]
    x2, y2, z2 = v2[0], v2[1], v2[2]
    x3, y3, z3 = v3[0], v3[1], v3[2]
    x1k, y1k, z1k = v1k[0], v1k[1], v1k[2]
    x2k, y2k, z2k = v2k[0], v2k[1], v2k[2]
    x3k, y3k, z3k = v3k[0], v3k[1], v3k[2]
    a1, a2, a3, b1, b2, b3, c1, c2, c3 = symbols('a1, a2, a3, b1, b2, b3, c1, c2, c3')
    system = [a1*x1 + a2*y1 + a3*z1 - x1k,
              b1*x1 + b2*y1 + b3*z1 - y1k,
              c1*x1 + c2*y1 + c3*z1 - z1k,
              a1*x2 + a2*y2 + a3*z2 - x2k,
              b1*x2 + b2*y2 + b3*z2 - y2k,
              c1*x2 + c2*y2 + c3*z2 - z2k,
              a1*x3 + a2*y3 + a3*z3 - x3k,
              b1*x3 + b2*y3 + b3*z3 - y3k,
              c1*x3 + c2*y3 + c3*z3 - z3k]
    symbols_to_found = (a1, a2, a3, b1, b2, b3, c1, c2, c3)
    answer = linsolve(system, symbols_to_found)
    print(answer)
    return np.array(answer.args[0], dtype=np.float32)


def gram_schmidt(V):
    U = np.zeros_like(V)
    for i in range(len(V)):
        u = V[i]
        for j in range(i):
            u -= np.dot(U[j], V[i]) * U[j]
        u /= np.linalg.norm(u)
        U[i] = u
    return U


test_img = plt.imread('test_face.jpg')
base_landmarks, base_blendshapes, base_matrixes = get_3d_facial_keypoints_large('base_face.png')
test_landmarks, test_blenshape, test_matrixes = get_3d_facial_keypoints_large('test_face.jpg')
print(base_landmarks[0], base_blendshapes[0], base_matrixes[0], sep='\n_______\n')
base_indexes = [4, 23, 253]
base_coordinates = []
test_coordinates = []
all_base_coordinates = []
all_test_coordinates = []
for i in range(len(base_landmarks[0])):
    all_base_coordinates.append([base_landmarks[0][i].x, base_landmarks[0][i].y, base_landmarks[0][i].z])
all_base_coordinates = np.float32(all_base_coordinates)
for i in range(len(test_landmarks[0])):
    all_test_coordinates.append([test_landmarks[0][i].x, test_landmarks[0][i].y, test_landmarks[0][i].z])
all_test_coordinates = np.float32(all_test_coordinates)
for i in base_indexes:
    base_coordinates.append([base_landmarks[0][i].x, base_landmarks[0][i].y, base_landmarks[0][i].z])
    test_coordinates.append([test_landmarks[0][i].x, base_landmarks[0][i].y, base_landmarks[0][i].z])
base_coordinates = np.float32(base_coordinates)
test_coordinates = np.float32(test_coordinates)
transition_matrix = np.reshape(find_transition_matrix(base_coordinates, test_coordinates), (3, 3))
print(transition_matrix)
new_basis = gram_schmidt(test_coordinates)
transposed_new_basis = transpose(new_basis)
invariant_new_basis = inv(new_basis)
invariant_transition_matrix = inv(transition_matrix)
transposed_transition_matrix = transpose(transition_matrix)
invariant_transposed_transition_matrix = inv(transposed_transition_matrix)
transposed_invariant_transition_matrix = transpose(invariant_transition_matrix)
plt.imshow(test_img)
for i in range(len(all_test_coordinates)):
    plt.scatter(all_test_coordinates[i][0] * test_img.shape[1], all_test_coordinates[i][1] * test_img.shape[0], c='red', marker='o')
plt.show()
print()
print(all_test_coordinates)
print(all_test_coordinates.shape)
print('_'*70)
for i in range(len(all_test_coordinates)):
    all_test_coordinates[i] = matmul(all_test_coordinates[i], invariant_new_basis)
plt.imshow(test_img)
for i in range(len(base_indexes)):
    plt.scatter(all_test_coordinates[base_indexes[i]][0] * test_img.shape[1], all_test_coordinates[base_indexes[i]][1] * test_img.shape[0], c='red', marker='o')
    plt.scatter(all_base_coordinates[base_indexes[i]][0] * test_img.shape[1], all_base_coordinates[base_indexes[i]][1] * test_img.shape[0], c='green', marker='o')
plt.show()
print(all_test_coordinates)
print(all_test_coordinates.shape)

