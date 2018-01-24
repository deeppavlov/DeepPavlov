"""
:class:`deeppavlov.models.model.Serializable` is an abstract base class that expresses the interface
for all models that can serialize data to a path.
"""

from abc import ABCMeta
from pathlib import Path
from urllib import request
import tarfile

from deeppavlov.core.common import paths
from deeppavlov.core.common.errors import ConfigError


class Serializable(metaclass=ABCMeta):
    """
    :attr: `_ser_dir` is a name of a serialization dir, can be default or set in json config
    :attr: `_ser_file` is a name of a serialization file (usually binary model file),
     can be default or set in json config
    :attr: `ser_path` is a path to model serialization dir or file (it depends on the model type).
     It is always an empty string and is ignored if it is not set in json config.
    """

    def __init__(self, ser_path=None, ser_dir=None, ser_file=None, url=None, **kwargs):
        self.url = url
        self._ser_dir = ser_dir
        self._ser_file = ser_file
        self.ser_path = self.get_ser_path(ser_path)

        if self.url:
            self.download()

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
            if p.is_dir():
                if p.name != self._ser_dir:
                    p = p / self._ser_dir
                    p.mkdir(parents=True, exist_ok=True)
            # elif p.is_file():
            #     pass
            # else:
            #     raise ConfigError("Provided ser_path doesn't exist!")

        return p

    def download(self):
        url = self.url
        print("Extracting files from url")
        local_filename, _ = request.urlretrieve(url)
        files = tarfile.open(local_filename, mode='r:gz')

        save_path = self.ser_path
        if not save_path.is_dir():
            save_path = save_path.parent
            if save_path.name == self._ser_dir:
                save_path = save_path.parent
        files.extractall(save_path)
