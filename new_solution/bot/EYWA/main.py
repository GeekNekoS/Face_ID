import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import dlib

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

from numpy import asarray


predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
desiredLeftEye = (0.25, 0.25)

# return face coordinates
def face_detection(img: np.array, min_detect_size: int = 30):
    index, faces = 0, detector(img, 1)
    while index != len(faces):
        if rect_to_bb(faces[index])[2] < min_detect_size:
            faces.pop(index)
        else: index += 1
    return faces
    
# change image size
def resize_image(img_array: np.array, img_size:int):
    down_points = (img_size, img_size)
    return cv2.resize(img_array, down_points, interpolation= cv2.INTER_LINEAR)
   

def extract_faces(img_path:str, min_detect_size:int, img_size:int=None, save_files_path:str=None) -> np.array:
    faces_list: list[np.array] = list()
    
    image: np.array = cv2.imread(img_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    
    for index, rect in enumerate(face_detection(gray_img, min_detect_size)):
        shape = predictor(gray_img, rect)
        shape = shape_to_np(shape)

        lStart, lEnd = (0, 1)
        rStart, rEnd = (2, 3)
            
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        desiredRightEyeX = 1.0 - desiredLeftEye[0]
        
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= img_size
        scale = desiredDist / dist
        
        eyesCenter = (int(leftEyeCenter[0] + rightEyeCenter[0]) // 2,
			int(leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        
        tX = img_size * 0.5
        tY = img_size * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        (w, h) = (img_size, img_size)
        output = cv2.warpAffine(image, M, (w, h),
			flags=cv2.INTER_CUBIC)
        

        if img_size:
            img = resize_image(output, img_size)
            
        if save_files_path:
            if not os.path.exists(save_files_path):
                 os.makedirs(save_files_path)
            
            save_path = os.path.join(save_files_path, f'{index}_' + os.path.basename(img_path))
            
            cv2.imwrite(save_path, img)
            
        faces_list.append(output)
        
    return faces_list


def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords


def get_faces_vector(faces):
    faces = [resize_image(face, 224) for face in faces]
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg')
    
    return model.predict(samples)

def compare_faces(face_vectors: np.array):
    return 1 - cosine(face_vectors[0], face_vectors[1])
