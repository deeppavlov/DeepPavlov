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

