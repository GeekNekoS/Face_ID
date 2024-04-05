import numpy as np


def find_faces_distance(face1_encoding: np.array, face2_encoding: np.array) -> float:
    if not face1_encoding or not face2_encoding:
        return float('inf')
    return np.linalg.norm(face1_encoding - face2_encoding)


def compare2faces(face1_encoding: np.array, face2_encoding: np.array, tolerance=0.6) -> bool:
    return find_faces_distance(face1_encoding, face2_encoding) <= tolerance