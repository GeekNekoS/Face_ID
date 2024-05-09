from face_recognition.detect_face import detect_face
from face_compare.compare2faces import Img2VecModel, cos_similarity
model = Img2VecModel()


def test(image_path1, image_path2):
    coordinates1 = detect_face(image_path1, image_path1)
    coordinates2 = detect_face(image_path2, image_path2)
    face1 = image_path1[coordinates1[1]:coordinates1[1]+coordinates1[3]+1, coordinates1[0]:coordinates1[0]+coordinates1[2]+1]
    face2 = image_path1[coordinates2[1]:coordinates2[1]+coordinates2[3]+1, coordinates2[0]:coordinates2[0]+coordinates2[2]+1]
    return cos_similarity(model.get_vec(face1), model.get_vec(face2))


result = test('../test_data/same1_1.jpg', '../test_data/same1_2.jpg')
print(result)
result = test('../test_data/same2_1.jpg', '../test_data/same2_2.jpg')
print(result)
result = test('../test_data/same1_1.jpg', '../test_data/same2_1')
print(result)