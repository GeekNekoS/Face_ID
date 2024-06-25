from new_solution.bot.EYWA.main import *
import zipfile

with open('datasets/data/new.txt', 'w') as new_txt:
    with open('datasets/data/identity_CelebA.txt', 'r') as old_text:
        with zipfile.ZipFile('datasets/data/data.zip', 'r') as zip_file:
            for _ in range(10**6):
                file_name, identifier = old_text.readline().split()
                with zip_file.open(file_name) as face:
                    faces = face_detection(face)
                    vector = get_faces_vector(faces[0])
                    new_txt.write(f'{file_name} {identifier} {vector}')