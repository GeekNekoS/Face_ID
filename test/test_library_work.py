from face_recognition.detect_face import detect_face
import cv2
from facial_key_points_detection.find_facial_key_points import find_facial_key_points
from face_alignment.face_alignment import ImageTransformation
import numpy as np


def connection_of_components(image_path: str, output_path: str, min_face_size: tuple[int, int] = (30, 30), model_path: str = '../CNNs/modelCNN(mae).h5', base_face_parts_coordinates_for_affine_transformation: np.ndarray | None = None):
    coordinates = detect_face(image_path, output_path, min_face_size)
    base_face = cv2.imread(output_path)[coordinates[0][1]:coordinates[0][1]+coordinates[0][3]+1, coordinates[0][0]:coordinates[0][0]+coordinates[0][2]+1, :]
    facial_key_points = find_facial_key_points(base_face, model_path)

    width = min_face_size[0]
    height = min_face_size[1]

    left_eye_coordinates = [facial_key_points[0]*width, facial_key_points[1]*height]
    nose_coordinates = [facial_key_points[20]*width, facial_key_points[21]*height]
    right_eye_coordinates = [facial_key_points[2]*width, facial_key_points[3]*height]
    affine_transformation = ImageTransformation(base_face)
    face_parts_coordinates = np.array([left_eye_coordinates, nose_coordinates, right_eye_coordinates]).astype(np.float32)
    affine_transformation.align_face(face_parts_coordinates, base_face_parts_coordinates_for_affine_transformation)

    # Функция нахождения вектора признаков
    # sign_vector = find_sing_vector(affine_transformation.transformed_image)

    # Функция сравнения двух векторов
