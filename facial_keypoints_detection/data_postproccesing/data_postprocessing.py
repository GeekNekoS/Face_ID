import json
import numpy as np


class ImageLabelsJSON:

    def __init__(self, file_path: str, image: np.ndarray[int, ...], labels: tuple[str, ...], image_name: str = 'img', values: tuple[float, ...] | None = None):
        self.path: str = file_path
        self.image = image
        self.image_name = image_name
        self.labels: tuple[str, ...] = labels
        self.values = values
        self.info = {}
        if values:
            self.make_json_file_with_ready_values()
        else:
            self.make_json_file()

    def make_json_file(self):
        self.info['name'] = self.image_name
        self.info['coordinates'] = {}
        for label in self.labels:
            self.info['coordinates'][label] = None
        with open(self.path, 'a') as json_file:
            json.dump(self.info, json_file)

    def make_json_file_with_ready_values(self):
        self.info['name'] = self.image_name
        self.info['coordinates'] = {}
        for i in range(len(self.labels)):
            self.info['coordinates'][self.labels[i]] = self.values[i]
        with open(self.path, 'a') as json_file:
            json.dump(self.info, json_file)

    def change_coordinates(self, new_coordinates):
        for i in range(len(self.labels)):
            self.info['coordinates'][self.labels[i]] = new_coordinates[i]
        with open(self.path, 'w') as json_file:
            json.dump(self.info, json_file)
