"""
:class:`deeppavlov.models.model.Trainable` is an abstract base class that expresses the interface
for all models that can be trained (ex. neural networks, scikit-learn estimators, gensim models,
etc.). All trainable models should inherit from this class.
"""

from abc import ABCMeta, abstractmethod
from pathlib import Path
from deeppavlov.core.common import paths


class Trainable(metaclass=ABCMeta):
    """
    :attr:`train_now` expresses a developer intent for whether a model as part of a pipeline
    should be trained in the current experiment run or not.
    """
    train_now = False
    _model_dir = ''
    _model_file = ''

    @property
    def model_path(self):
        return Path(paths.USR_PATH).joinpath(self._model_dir, self._model_file)

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

    def save(self):
        pass

    def load(self):
        pass
