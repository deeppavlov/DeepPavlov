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
    model_path = ''

    @property
    def model_path_(self) -> Path:
        if not self.model_path:
            print("model path None")
            return Path(paths.USR_PATH).joinpath(self._model_dir, self._model_file)
        else:
            print("model path not None")
            return Path(self.model_path)

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

    def make_dir(self):
        if not self.model_path_.parent.exists():
            Path.mkdir(self.model_path_.parent)
