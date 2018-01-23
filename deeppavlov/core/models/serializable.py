"""
:class:`deeppavlov.models.model.Serializable` is an abstract base class that expresses the interface
for all models that can serialize data to a path.
"""

from abc import ABCMeta
from pathlib import Path

from deeppavlov.core.common import paths


class Serializable(metaclass=ABCMeta):
    """
    :attr: `_ser_dir` is a name of a serialization dir, can be default or set in json config
    :attr: `_ser_file` is a name of a serialization file (usually binary model file),
     can be default or set in json config
    :attr: `ser_path` is a path to model serialization dir or file (it depends on the model type).
     It is always an empty string and is ignored if it is not set in json config.
    """

    def __init__(self, ser_path=None, ser_dir=None, ser_file=None, **kwargs):
        self._ser_dir = ser_dir
        self._ser_file = ser_file
        self.ser_path = self.get_ser_path(ser_path)

    def __new__(cls, *args, **kwargs):
        if cls is Serializable:
            raise TypeError(
                "TypeError: Can't instantiate abstract class {} directly".format(cls.__name__))
        return object.__new__(cls)

    def get_ser_path(self, ser_path):
        if not ser_path:
            p = paths.USR_PATH
            if self._ser_dir:
                p = p / self._ser_dir
                p.mkdir(parents=True, exist_ok=True)
            if self._ser_file:
                p = p / self._ser_file
        else:
            p = Path(ser_path)
        return p
