from typing import List, Dict

from common.registrable import Registrable


class DatasetReader(Registrable):
    """
    A ``DatasetReader`` reads data from some location and constructs a :class:`Dataset`.
    """

    def read(self, file_path: str) -> List[Dict]:
        """
        Reads a file from a path and returns data as list. The list should consist of training
        instances as dicts, each of which has `data` and `target` keys.
        Example:
             [ {'data': 'good morning!', 'target': 'hello what can i help you with today'},
             {'data': 'i'd like to book a table with italian food', 'target': 'i'm on it'}]
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Dict) -> 'DatasetReader':
        """
        Static method that constructs the dataset reader described by ``params`` in a config file.
        """
        # choice = params.pop_choice('type', cls.list_available())
        # TODO check if available in registry
        name = params['name']
        return cls.by_name(name).from_params(params)
