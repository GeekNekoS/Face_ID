import os
import keras
from yandex_disk.import_file_from_yandex_disk import import_file


def import_model_from_yandex_disk(yadisk_url_download: str, yadisk_token: str, yadisk_path: str, path: str, custom_objects: dict = None, save_model: bool = False):
    """
    Imports model from Yandex disk.
    :param yadisk_url_download: str, url for download file from yandex disk
    :param yadisk_token: str, yandex disk api token
    :param save_model: bool, save file with model in project or not [optional]
    :param yadisk_path: str, file path in yandex disk
    :param custom_objects: dict, custom objects for load models [optional]
    :param path: str, file path in project
    :return: CNN model
    """
    import_file(yadisk_url_download, yadisk_token, yadisk_path, path)
    if custom_objects is not None:
        model = keras.models.load_model(path, custom_objects)
    else:
        model = keras.models.load_model(path)
    if not save_model:
        os.remove(path)
    return model