import mediapipe as mp
import mediapipe.tasks.python.vision
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result: mediapipe.tasks.python.vision.FaceLandmarkerResult) -> None:
    """
    Draw face mesh.
    :param rgb_image: ndarray, three-dimensional array from image
    :param detection_result: face landmarks detection result
    :return: None
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    for idx in range(len(face_landmarks_list)):
      face_landmarks = face_landmarks_list[idx]
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


def get_3d_facial_key_points_large(image_path: str) -> mediapipe.tasks.python.vision.FaceLandmarkerResult:
    """
    Return face mesh, blendshape, transformation matrix of input image
    :param image_path: str, image path
    :return: FaceLandmarkerResult
    """
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    image = mediapipe.Image(image_path)
    img = mp.Image.create_from_file(image)
    results = detector.detect(img)
    return results
