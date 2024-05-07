import keras
import requests
import os
from dotenv import load_dotenv
load_dotenv('../yandex.env')


def import_CNN(save_model: bool = False):
    """
    Imports model for facial key points detection from Yandex disk
    :return: None
    """
    headers = {'Authorization': f'OAuth {os.getenv("YANDEX_DISK_API_TOKEN")}'}
    url = os.getenv('YANDEX_DISK_MODEL')
    response = requests.get(url, headers=headers)
    file_url = response.json()['href']
    file_response = requests.get(file_url)
    with open('modelCNN(mae).h5', 'wb') as file:
        file.write(file_response.content)
        model = keras.models.load_model('modelCNN(mae).h5', {'mae': keras.losses.mean_absolute_error, 'huber': keras.losses.huber, 'log_cosh': keras.losses.log_cosh})
    if not save_model:
        os.remove('modelCNN(mae).h5')
    return model
