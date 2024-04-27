import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageTransformation:
    # def __init__(self, file_path: str):
    #     self.path = file_path
    #     self.image = plt.imread(self.path)
    #     self.transformed_image = None

    def __init__(self, image: np.ndarray[int, ...]):
        self.image = image
        self.transformed_image = None

    def face_alignment(self, face_parts_coordinates: np.ndarray[float, ...], base_face_parts_coordinates: np.ndarray[float, ...]=None):
        """
        make transformation using coordinates
        face_parts_coordinates: np.ndarray[[l.eye], [nose], [r.eye]]
        """
        if base_face_parts_coordinates is None:
            base_face_parts_coordinates = np.array([[27.2, 35.0], [50.0, 55.0], [70.4, 35.0]]).astype(np.float32)
        rows, columns, shape = self.image.shape
        matrix = cv2.getAffineTransform(face_parts_coordinates, base_face_parts_coordinates)
        result = cv2.warpAffine(self.image, matrix, (columns, rows))
        self.transformed_image = result

    def draw_transformed_image(self):
        plt.imshow(self.transformed_image)
        plt.show()

    def draw_start_image(self):
        plt.imshow(self.image)
        plt.show()


# path3 = 'images/examples/resized_img_to_transformation.jpg'
# coord = np.array([[35.0, 40.0], [53.0, 57.0], [64.0, 37.0]]).astype(np.float32)
# example = ImageTransformation(path3)
# example.face_alignment(coord)
# example.draw_start_image()
# example.draw_transformed_image()

