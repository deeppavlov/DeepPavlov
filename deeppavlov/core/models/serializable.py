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

    _model_dir = ''
    _model_file = ''
    model_path = ''

    @property
    def model_path_(self) -> Path:
        if not self.model_path:
            return Path(paths.USR_PATH).joinpath(self._model_dir, self._model_file)
        else:
            return Path(self.model_path)
