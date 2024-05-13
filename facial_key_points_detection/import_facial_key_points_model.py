import os
from yandex_disk.import_model_from_yandex_disk import import_model_from_yandex_disk
from dotenv import load_dotenv
load_dotenv('../yandex.env')


def import_model(custom_objects: dict, yadisk_url_download: str = os.getenv('YANDEX_DISK_DOWNLOAD'), yadisk_token: str = os.getenv('YANDEX_DISK_API_TOKEN'), yadisk_path: str = os.getenv('FACIAL_KEY_POINTS_MODEL_PATH'), path: str = 'facial_key_points_model.h5', save_model: bool = False):
    """
    Imports model for facial key points detection from Yandex disk.
    :param custom_objects: dict, custom objects for load models
    :param yadisk_url_download: str, url for download file from yandex disk [optional]
    :param yadisk_token: str, yandex disk api token [optional]
    :param save_model: bool, save file with model in project or not [optional]
    :param yadisk_path: str, file path in yandex disk [optional]
    :param path: str, file path in project [optional]
    :return: CNN model
    """
    model = import_model_from_yandex_disk(yadisk_url_download, yadisk_token, yadisk_path, path, save_model, custom_objects)
    return model
