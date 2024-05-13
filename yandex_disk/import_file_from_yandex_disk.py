import requests


def import_file(yadisk_url_download: str, yadisk_token: str, yadisk_path: str, path: str) -> None:
    """
    Download file from yandex disk
    :param yadisk_url_download: str, url for download file from yandex disk
    :param yadisk_token: str, yandex disk api token
    :param yadisk_path: str, file path in yandex disk
    :param path: str, file path in project
    :return: None
    """
    headers = {'Authorization': f'OAuth {yadisk_token}'}
    url = yadisk_url_download + yadisk_path
    response = requests.get(url, headers=headers)
    file_url = response.json()['href']
    file_response = requests.get(file_url)
    with open(path, 'wb') as file:
        file.write(file_response.content)
