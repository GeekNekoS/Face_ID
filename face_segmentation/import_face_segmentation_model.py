import os
from import_model import import_model
from dotenv import load_dotenv
load_dotenv('../yandex.env')


def import_U_net(custom_objects: dict, yadisk_path: str = os.getenv('FACE_SEGMENTATION_MODEL_PATH'), path: str = 'face_segmentation_model.h5', save_model: bool = False):
    """
    Import U_net model.
    :param yadisk_path: str, file path in yandex disk
    :param path: str, file path in project
    :param save_model: bool, save file with model in project or not
    :return: U_net model
    """
    model = import_model(yadisk_path, path, custom_objects, save_model)
    return model
