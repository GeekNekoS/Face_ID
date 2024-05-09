import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageTransformation:
    """
    Class for transformation image.
    """
    def __init__(self, image: np.ndarray, transformed_image: np.ndarray | None = None) -> None:
        """
        :param image: ndarray, three-dimensional array from image
        :param transformed_image: ndarray, three-dimensional array from aligned image [optional]
        :return: None
        """
        self.image: np.ndarray = image
        self.transformed_image: np.ndarray | None = transformed_image

    def align_face(self, face_parts_coordinates: np.ndarray, base_face_parts_coordinates: np.ndarray | None = None) -> None:
        """
        Make transformation using normalized coordinates.
        :param face_parts_coordinates: np.ndarray[[left eye], [nose], [right eye]]
        :param base_face_parts_coordinates: 3 base face key points coordinates
        :return: None
        """
        if base_face_parts_coordinates is None:
            base_face_parts_coordinates = np.array([[0.28, 0.36], [0.52, 0.57], [0.73, 0.36]]).astype(np.float32)
        rows, columns, shape = self.image.shape
        matrix = cv2.getAffineTransform(face_parts_coordinates, base_face_parts_coordinates)
        result = cv2.warpAffine(self.image, matrix, (columns, rows))
        self.transformed_image = result

    def draw_transformed_image(self):
        """
        Draw aligned face.
        :return: None
        """
        if self.transformed_image is not None:
            plt.imshow(self.transformed_image)
            plt.show()

    def draw_start_image(self):
        """
        Draw base face.
        :return: None
        """
        plt.imshow(self.image)
        plt.show()
