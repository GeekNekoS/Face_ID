import cv2

def detect_face(image_path, output_path):
    # Загрузка каскадного классификатора для детекции лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Преобразование изображения в оттенки серого (для детекции лиц работает лучше)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детекция лиц на изображении
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    x,y,w,h = faces[0]
    roi_color = image[y:y+h, x:x+w]
    # resize_image = cv2.resize(roi_color, (96,96))
    resize_image = roi_color
    cv2.imwrite(output_path, resize_image)

    return faces


# Пример использования
# input_image_path = "pavel.jpg"
# output_image_path = "output.jpg"
# print(input_image_path)

# faces = detect_face(input_image_path, output_image_path)
# if len(faces) > 0:
#     print("Лица обнаружены:")
#     for i, (x, y, w, h) in enumerate(faces, 1):
#         print(f"Лицо {i}: координаты ({x}, {y}), ширина {w}, высота {h}")
# else:
#     print("Лица не обнаружены на изображении.")


# Этот код использует каскадный классификатор Хаара из OpenCV для детекции лиц на изображении. Учитывая указанную версию
# OpenCV (4.9.0.80), используется новый способ загрузки классификатора с помощью cv2.data.haarcascades
