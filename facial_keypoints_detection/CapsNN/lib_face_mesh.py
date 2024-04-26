import cv2
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils
# cap = cv2.VideoCapture(0)
while True:
    img = cv2.imread('../CNNs/face.jpg')
    results = face_mesh.process(img)
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, landmarks, mp_face_mesh.FACEMESH_TESSELATION, mp_draw.DrawingSpec((0, 255, 0), 1, 1), mp_draw.DrawingSpec((0, 0, 255), 1, 1))
    cv2.imshow("Face Mesh", img)
    cv2.waitKey(1)