import cv2
from face_alignment import ImageTransformation


def png_test(image_path, face_parts_coordinates):
    image = cv2.imread(image_path)
    image = ImageTransformation(image)
    image.align_face(face_parts_coordinates)
    return image.transformed_image


def jpg_test(image_path, face_parts_coordinates):
    image = cv2.imread(image_path)
    image = ImageTransformation(image)
    image.align_face(face_parts_coordinates)
    return image.transformed_image
