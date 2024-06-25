import zipfile
import random
import time
from face_recognition.detect_face import detect_face


with zipfile.ZipFile('datasets/data/data.zip', 'r') as zip_file:
    time1 = 0
    for i in range(100):
        index = random.randrange(1, 10**6)
        str_index = ('0' * (6 - len(str(index)))) + str(index)
        with zip_file.open(f'{str_index}.png') as face:
            start_time = time.time()
            faces = detect_face(face, None)
            time1 += (time.time() - start_time)
    print(time1)