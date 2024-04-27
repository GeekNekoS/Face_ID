import cv2


def detect_face(image_path, output_path, min_size=(30, 30)):
    """
    Функция для детекции лиц на изображении с возможностью указания минимального размера лица.

    :param image_path: Путь к входному изображению.
    :param output_path: Путь для сохранения изображения с отмеченными лицами.
    :param min_size: Кортеж, задающий минимальный размер лица для обнаружения.
    """
    # Загрузка каскадного классификатора для детекции лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Преобразование изображения в оттенки серого (для детекции лиц работает лучше)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детекция лиц на изображении
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_path, image)

    return faces


# Пример использования
# input_image_path = "data/collected_images/6976fd7e-ee5f-11ee-a961-f42679e0f8f2.jpg"
input_image_path = "data/nekos_images/test_5.jpg"

output_image_path = "output.jpg"
min_size = (65, 65)  # Установка минимального размера лица

faces = detect_face(input_image_path, output_image_path, min_size)
if len(faces) > 0:
    print("Лица обнаружены:")
    for i, (x, y, w, h) in enumerate(faces, 1):
        print(f"Лицо {i}: координаты ({x}, {y}), ширина {w}, высота {h}")
else:
    print("Лица не обнаружены на изображении.")
