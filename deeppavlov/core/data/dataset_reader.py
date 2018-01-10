from abc import abstractmethod
from typing import List


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
    @abstractmethod
    def save_vocab(data, ser_dir):
        """
        Extract single words from data and save them to a serialization dir.
        :param data: dataset
        :param ser_dir specified by user serialization dir
        """
        pass
