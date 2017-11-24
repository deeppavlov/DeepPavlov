"""This is a base abstract class for other abstract model classes. Your model should be inherited
from one of the particluar classes, :class:`TensorflowModel`, :class:`SklearnModel`,
:class:`NonTrainableModel` """
from inspect import getfullargspec
from typing import List, Dict, Type
from deeppavlov.common.registry import _REGISTRY


class Model:

    def infer(self, data):
        """
        Predict data.
        :param data: any type of input data
        :return: any type of predicted values
        """
        raise NotImplementedError
