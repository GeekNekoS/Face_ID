from image.single_image_preprocessing import preprocess_image
import cv2


def convert_image(image, new_size=(512, 512)):
    converted_image = preprocess_image(image, new_size, cv2.COLOR_BGR2RGB)
    return converted_image