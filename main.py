import settings
import easyocr


def text_recognition(file_path):
    reader = easyocr.Reader(settings.LANGUAGES_TO_RECOGNIZE)
    text = reader.readtext(file_path, detail=0, batch_size=1)
    return text


def main():
    file_path = settings.IMAGES_DIR + "\\text.jpg"
    text = text_recognition(file_path=file_path)
    print(text)


if __name__ == "__main__":
    main()
