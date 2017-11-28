from typing import Dict, List
from inspect import getfullargspec

from deeppavlov.common.registry import _REGISTRY


class DatasetReader:
    """
    A ``DatasetReader`` reads data from some location and constructs a dataset.
    """

    def read(self, file_path: str) -> List:
        """
        Read a file from a path and returns data as list with training instances.
        """
        raise NotImplementedError

    @staticmethod
    def save_vocab(data, ser_dir):
        """
        Extract single words from data and save them to a serialization dir.
        :param data: dataset
        :param ser_dir specified by user serialization dir
        """
        pass
