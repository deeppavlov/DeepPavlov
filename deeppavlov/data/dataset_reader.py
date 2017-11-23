from typing import Dict, List
from inspect import getfullargspec


class DatasetReader:
    """
    A ``DatasetReader`` reads data from some location and constructs a dataset.
    """

    def read(self, file_path: str) -> List:
        """
        Reads a file from a path and returns data as list with training instances.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Dict) -> 'DatasetReader':
        """
        Take config params and build a class instance. If params are not set in json config, default
        params are used.
        :param params: class params from json config
        :return: a class instance
        """

        signature_params = getfullargspec(cls.__init__).args[1:]
        param_dict = {}
        for sp in signature_params:
            try:
                param_dict[sp] = params[sp]
            except KeyError:
                pass

        return cls(**param_dict)
