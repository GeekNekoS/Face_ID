import os
import time
import uuid
import cv2


IMAGES_PATH = os.path.join('data', 'collected_images')
number_images = 30

cap = cv2.VideoCapture(0)  # 0 - номер вашей видеокамеры
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
