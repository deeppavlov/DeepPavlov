"""
:class:`deeppavlov.models.model.Trainable` is an abstract base class that expresses the interface
for all models that can be trained (ex. neural networks, scikit-learn estimators, gensim models,
etc.). All trainable models should inherit from this class.
"""

from abc import ABCMeta, abstractmethod


class Trainable(metaclass=ABCMeta):
    """
    :attr:`train_now` expresses a developer intent for whether a model as part of a pipeline
    should be trained in the current experiment run or not.
    """
    train_now = False

    @abstractmethod
    def train(self, features, *args):
        """
        Train a model.
        :param features: any type of input data passed for training
        :param args: all needed params for training
        As a result of training, the model should be saved to user dir defined at
        deeppavlov.common.paths.USR_PATH. Remember that a particular path is assigned in runtime.
        """
        pass
