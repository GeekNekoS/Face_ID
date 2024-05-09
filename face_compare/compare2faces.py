import cv2
import numpy as np
from tensorflow.keras.applications import resnet50
from tensorflow.keras.models import Model
from numpy.linalg import norm


def cos_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = norm(vec1)
    norm_vec2 = norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


class Img2VecModel(object):
    """
    Model class for extracting a vector of facial features.
    """
    def __init__(self):
        """
        Creates model for extracting a vector of facial features.
        """
        model = resnet50.ResNet50(weights='imagenet')
        layer_name = 'avg_pool'
        self.intermediate_layer_model = Model(inputs=model.input, 
                                              outputs=model.get_layer(layer_name).output)
        
    def get_vec(self, image):
        """
        Gets a vector embedding from an image.
        :param image: array from image
        :returns: numpy ndarray
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.expand_dims(gray, axis=0)
        preprocessed_input = resnet50.preprocess_input(gray)
        intermediate_output = self.intermediate_layer_model.predict(preprocessed_input)
        return intermediate_output[0]
