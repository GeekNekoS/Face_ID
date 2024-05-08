import os
import keras
from import_from_yandex_disk import import_from_yadisk


def import_model(yadisk_path: str, path: str, custom_objects: dict, save_model: bool = False):
    """
    Imports model for facial key points detection from Yandex disk
    :param save_model: bool, save file with model in project or not
    :param yadisk_path: str, file path in yandex disk
    :param path: str, file path in project
    :return: CNN model
    """
    import_from_yadisk(yadisk_path, path)
    model = keras.models.load_model(path, custom_objects)
    if not save_model:
        os.remove(path)
    return model