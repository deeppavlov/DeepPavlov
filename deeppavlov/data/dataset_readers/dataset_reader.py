from typing import Dict


class DatasetReader:
    """
    A ``DatasetReader`` reads data from some location and constructs a dataset.
    """
    @staticmethod
    def read(data_path: str, *args, **kwargs) -> Dict:
        """
        Read a file from a path and returns data as list with training instances.
        """
        raise NotImplementedError
