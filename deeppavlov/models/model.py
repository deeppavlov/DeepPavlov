"""This is a base abstract class for other abstract model classes. Your model should be inherited
from one of the particluar classes, :class:`TensorflowModel`, :class:`SklearnModel`,
:class:`NonTrainableModel` """

from typing import List, Dict, Type


class Model:
    def __init__(self, models: List = None, params: Dict = None):
        """
        :param models: any nested models
        :param params: hyperparameters/parameters of the model
        """
        self._models = models
        self._params = params

    def infer(self, data):
        """
        Predict data.
        :param data: any type of input data
        :return: any type of predicted values
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Dict) -> Type:
        """
          Static method that constructs the dataset reader described by ``params`` in a config file.
          ``params`` arg is what comes from the json config.

          Example:
              signature_params = ['param1', 'param2']
              param_dict = {}
              for sp in signature_params:
                  try:
                      param_dict[sp] = params[sp]
                  except KeyError:
                      pass

              return TestModel(**param_dict)
        """
        raise NotImplementedError
