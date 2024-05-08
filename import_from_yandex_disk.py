import requests
import os
from dotenv import load_dotenv
load_dotenv('yandex.env')


def import_from_yadisk(yadisk_path: str, path: str) -> None:
    """
    Download file from yandex disk
    :param yadisk_path: str, file path in yandex disk
    :param path: str, file path in project
    :return: None
    """
    headers = {'Authorization': f'OAuth {os.getenv("YANDEX_DISK_API_TOKEN")}'}
    url = os.getenv('YANDEX_DISK_DOWNLOAD') + yadisk_path
    response = requests.get(url, headers=headers)
    file_url = response.json()['href']
    file_response = requests.get(file_url)
    with open(path, 'wb') as file:
        file.write(file_response.content)
