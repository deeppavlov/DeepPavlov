from typing import Dict, List
from inspect import getfullargspec

from deeppavlov.common.registry import _REGISTRY


class DatasetReader:
    """
    A ``DatasetReader`` reads data from some location and constructs a dataset.
    """

    def read(self, file_path: str) -> List:
        """
        Reads a file from a path and returns data as list with training instances.
        """
        raise NotImplementedError