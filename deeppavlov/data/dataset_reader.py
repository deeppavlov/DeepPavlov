from typing import Dict, Type


class DatasetReader:
    """
    A ``DatasetReader`` reads data from some location and constructs a dataset.
    """

    def read(self, file_path: str):
        """
        Reads a file from a path and returns data as list.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Dict) -> Type:
        """
        Static method that constructs the dataset reader described by ``params`` in a config file.
        ``params`` arg is what comes from the json config.

        Example:
            signature_params = ['param1', 'param2']
            param_dict = {}
            for sp in signature_params:
                try:
                    param_dict[sp] = params[sp]
                except KeyError:
                    pass

            return TestReader(**param_dict)
        """
        raise NotImplementedError
