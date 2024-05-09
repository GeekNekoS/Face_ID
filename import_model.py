import os
import keras
from import_from_yandex_disk import import_from_yadisk


def import_model(yadisk_path: str, path: str, custom_objects: dict = None, save_model: bool = False):
    """
    Imports model from Yandex disk
    :param save_model: bool, save file with model in project or not
    :param yadisk_path: str, file path in yandex disk
    :param custom_objects: dict, custom objects for load models [optional]
    :param path: str, file path in project [optional]
    :return: CNN model
    """
    import_from_yadisk(yadisk_path, path)
    if custom_objects is not None:
        model = keras.models.load_model(path, custom_objects)
    else:
        model = keras.models.load_model(path)
    if not save_model:
        os.remove(path)
    return model