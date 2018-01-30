"""
:class:`deeppavlov.models.model.Serializable` is an abstract base class that expresses the interface
for all models that can serialize data to a path.
"""

from abc import ABCMeta
from pathlib import Path
from warnings import warn


class Serializable(metaclass=ABCMeta):
    """
    :attr: `_ser_dir` is a name of a serialization dir, can be default or set in json config
    :attr: `_ser_file` is a name of a serialization file (usually binary model file),
     can be default or set in json config
    :attr: `ser_path` is a path to model serialization dir or file (it depends on the model type).
     It is always an empty string and is ignored if it is not set in json config.
    """

    def __init__(self, save_path, load_path=None, **kwargs):

        if save_path:
            self.save_path = Path(save_path)
            self.save_path.parent.mkdir(exist_ok=True)
        else:
            self.save_path = None

        mode = kwargs.get('mode', 'infer')

        if load_path:
            self.load_path = Path(load_path)
            if mode != 'train' and self.load_path != self.save_path:
                warn("Load path '{}' differs from save path '{}' in '{}' mode for {}."
                     .format(self.load_path, self.save_path, mode, self.__class__.__name__))
        elif mode != 'train' and self.save_path:
            self.load_path = self.save_path
            warn("No load path is set for {} in '{}' mode. Using save path instead"
                 .format(self.__class__.__name__, mode))
        else:
            self.load_path = None
            warn("No load path is set for {}!".format(self.__class__.__name__))

    def __new__(cls, *args, **kwargs):
        if cls is Serializable:
            raise TypeError(
                "TypeError: Can't instantiate abstract class {} directly".format(cls.__name__))
        return object.__new__(cls)
