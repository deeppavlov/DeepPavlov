"""This is a base abstract class for other abstract model classes. Your model should be inherited
from one of the particluar classes, :class:`TensorflowModel`, :class:`SklearnModel`,
:class:`NonTrainableModel` """

from typing import List


class Model:
    def __init__(self, config_path: str, models: List = None):
        """
        :param config_path: encoded JSON path to traverse the origin of the model
        :param models: any nested models
        """
        self.config_path = config_path
        self.models = models

    def predict(self, data):
        """
        Predict data.
        :param data: any type of input data
        :return: any type of predicted values
        """
        raise NotImplementedError

    def save(self):
        """
        Save model to file.
        """
        raise NotImplementedError

    def load(self):
        """
        Load model from file.
        """
        raise NotImplementedError
