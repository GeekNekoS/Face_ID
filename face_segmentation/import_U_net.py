import os
from import_model import import_model_from_yadisk
from dotenv import load_dotenv
load_dotenv('../yandex.env')


def import_model(custom_objects: dict, yadisk_url_download: str = os.getenv('YANDEX_DISK_DOWNLOAD'), yadisk_token: str = os.getenv('YANDEX_DISK_API_TOKEN'), yadisk_path: str = os.getenv('FACE_SEGMENTATION_MODEL_PATH'), path: str = 'face_segmentation_model.h5', save_model: bool = False):
    """
    Import U_net model.
    :param custom_objects: dict, custom objects for load models
    :param yadisk_url_download: str, url for download file from yandex disk [optional]
    :param yadisk_token: str, yandex disk api token [optional]
    :param yadisk_path: str, file path in yandex disk [optional]
    :param path: str, file path in project [optional]
    :param save_model: bool, save file with model in project or not [optional]
    :return: U_net model
    """
    model = import_model_from_yadisk(yadisk_url_download, yadisk_token, yadisk_path, path, custom_objects, save_model)
    return model
