"""
:class:`deeppavlov.models.model.Trainable` is an abstract base class that expresses the interface
for all models that can be trained (ex. neural networks, scikit-learn estimators, gensim models,
etc.). All trainable models should inherit from this class.
"""

from abc import abstractmethod

from .serializable import Serializable


class Trainable(Serializable):
    """
    :attr:`train_now` expresses a developer intent for whether a model as part of a pipeline
    should be trained in the current experiment run or not.
    """

    def __init__(self, train_now=False, **kwargs):
        mode = kwargs.get('mode', None)
        if mode == 'train':
            self.train_now = train_now
        else:
            self.train_now = False
        super().__init__(**kwargs)

    @abstractmethod
    def train(self, data, *args, **kwargs):
        """
        Train a model.
        :param data: any type of input data passed for training
        :param args: all needed params for training
        As a result of training, the model should be saved to user dir defined at
        deeppavlov.common.paths.USR_PATH. A particular path is assigned in runtime.
        """
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass
