import cv2


def face_alignment(face_img_path, face_parts_coordinates, base_face_parts_coordinates):
    img = cv2.imread(face_img_path)
    rows, columns, shape = img.shape
    matrix = cv2.getAffineTransform(face_parts_coordinates, base_face_parts_coordinates)
    result = cv2.warpAffine(img, matrix, (columns, rows))
    return result
