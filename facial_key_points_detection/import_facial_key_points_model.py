import os
from import_model import import_model
from dotenv import load_dotenv
load_dotenv('../yandex.env')


def import_CNN(custom_objects: dict, yadisk_path: str = os.getenv('FACIAL_KEY_POINTS_MODEL_PATH'), path: str = 'facial_key_points_model.h5', save_model: bool = False):
    """
    Imports model for facial key points detection from Yandex disk.
    :param save_model: bool, save file with model in project or not
    :param yadisk_path: str, file path in yandex disk
    :param path: str, file path in project
    :return: CNN model
    """
    model = import_model(yadisk_path, path, custom_objects, save_model)
    return model
