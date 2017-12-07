from typing import Dict


class DatasetReader:
    """
    A ``DatasetReader`` reads data from some location and constructs a dataset.
    """

    def read(self, file_path: str) -> Dict:
        """
        Read a file from a path and returns data as list with training instances.
        """
        raise NotImplementedError
