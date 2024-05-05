import cv2
import numpy as np

from tensorflow.keras.applications import resnet50
from tensorflow.keras.models import Model
from numpy.linalg import norm


# def find_faces_distance(face1_encoding: np.array, face2_encoding: np.array) -> float:
#     return np.linalg.norm(face1_encoding - face2_encoding)


# def compare2faces(face1_encoding: np.array, face2_encoding: np.array, tolerance: float = 0.6) -> bool:
#     return find_faces_distance(face1_encoding, face2_encoding) <= tolerance

def cos_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = norm(vec1)
    norm_vec2 = norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


class Img2Vec(object):
    def __init__(self):
        model = resnet50.ResNet50(weights='imagenet')
        layer_name = 'avg_pool'
        self.intermediate_layer_model = Model(inputs=model.input, 
                                              outputs=model.get_layer(layer_name).output)
        
    def get_vec(self, image_path):
        """ Gets a vector embedding from an image.
        :param image_path: path to image on filesystem
        :returns: numpy ndarray
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        cv2.imshow(gray)
        x = image.img_to_array(gray)
        x = np.expand_dims(x, axis=0)
        x = resnet50.preprocess_input(x)
        intermediate_output = self.intermediate_layer_model.predict(x)
        
        return intermediate_output[0]
    
    
# img2vec = Img2Vec()
# cat1_vec = np.array(img2vec.get_vec('output.jpg') )
# cat2_vec = np.array(img2vec.get_vec('output2.jpg') )

# print(cos_similarity(cat1_vec, cat2_vec))