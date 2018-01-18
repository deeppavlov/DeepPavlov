"""
:class:`deeppavlov.models.model.Serializable` is an abstract base class that expresses the interface
for all models that can serialize data to a path.
"""

from abc import ABCMeta
from pathlib import Path

from deeppavlov.core.common import paths


class Serializable(metaclass=ABCMeta):
    """
    :attr: `_model_dir` is a name of a serialization dir, can be default or set in json config
    :attr: `_model_file` is a name of a serialization file (usually binary model file),
     can be default or set in json config
    :attr: `model_path` is a path to model serialization dir or file (it depends on the model type).
     It is always an empty string and is ignored if it is not set in json config.
    """

    def __init__(self, model_path=None, model_dir=None, model_file=None, *args, **kwargs):
        self._model_dir = model_dir
        self._model_file = model_file
        self.model_path = self.get_model_path(model_path)

    def __new__(cls, *args, **kwargs):
        if cls is Serializable:
            raise TypeError(
                "TypeError: Can't instantiate abstract class {} directly".format(cls.__name__))
        return object.__new__(cls)

    def get_model_path(self, model_path):
        if not model_path:
            p = paths.USR_PATH
            if self._model_dir:
                p = p / self._model_dir
                p.mkdir(parents=True, exist_ok=True)
            if self._model_file:
                p = p / self._model_file
        else:
            p = Path(model_path)
        return p
