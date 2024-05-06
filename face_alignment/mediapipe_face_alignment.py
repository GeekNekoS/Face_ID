import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_aligner
base_options = python.BaseOptions(model_asset_path='../facial_key_points_detection/FaceMesh/face_landmarker.task')


def align_face(image_path: str) -> np.ndarray:
    """
    Face aligning with the mediapipe.
    :param image_path: str, image path
    :return: ndarray, array from image
    """
    options = vision.FaceAlignerOptions(base_options=base_options)
    FaceAligner = face_aligner.FaceAligner.create_from_options(options)
    img = mp.Image.create_from_file(image_path)
    result = FaceAligner.align(img)
    return result.numpy_view()
