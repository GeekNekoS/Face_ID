import json
import numpy as np


class ImageLabelsJSON:
    """
    Сlass for working with a json file containing all the information about the facial key points and their coordinates.
    """
    def __init__(self, file_path: str, image: np.ndarray[int, ...], labels: tuple[str, ...], image_name: str = 'img', values: np.ndarray | None = None) -> None:
        """
        Automatically creates a new json file or overwrites an existing one
        :param file_path: str, path to json file
        :param image: ndarray[int], array from image
        :param labels: tuple[str, ...], facial key points names
        :param image_name: str, image name [optional]
        :param values: tuple[float, ...], facial key points values [optional]
        :return: None
        """
        self.path: str = file_path
        self.image: np.ndarray[int, ...] = image
        self.image_name: str = image_name
        self.labels: tuple[str, ...] = labels
        self.values: np.ndarray[float, ...] = values
        self.info: dict = {}
        if values:
            self.make_json_file_with_ready_values()
        else:
            self.make_json_file()

    def make_json_file(self) -> None:
        """
        Creates a new json file without facial key points values or overwrites an existing one.
        :return: None
        """
        self.info['name'] = self.image_name
        self.info['coordinates'] = {}
        for label in self.labels:
            self.info['coordinates'][label] = None
        with open(self.path, 'w') as json_file:
            json.dump(self.info, json_file)

    def make_json_file_with_ready_values(self) -> None:
        """
        Creates a new json file with facial key points values or overwrites an existing one.
        :return: None
        """
        self.info['name'] = self.image_name
        self.info['coordinates'] = {}
        for i in range(len(self.labels)):
            self.info['coordinates'][self.labels[i]] = self.values[i]
        with open(self.path, 'w') as json_file:
            json.dump(self.info, json_file)

    def change_coordinates(self, new_coordinates: tuple[float, ...]) -> None:
        """
        Overwrites the json file with new data.
        :param new_coordinates: tuple[float, ...] facial key points values
        :return: None
        """
        for i in range(len(self.labels)):
            self.info['coordinates'][self.labels[i]] = new_coordinates[i]
        with open(self.path, 'w') as json_file:
            json.dump(self.info, json_file)
