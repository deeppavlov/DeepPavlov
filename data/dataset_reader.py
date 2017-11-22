from typing import List, Dict, Type


class DatasetReader:
    """
    A ``DatasetReader`` reads data from some location and constructs a :class:`Dataset`.
    """

    def read(self, file_path: str) -> List[Dict]:
        """
        Reads a file from a path and returns data as list. The list should consist of training
        instances as dicts, each of which has `context` and `response` keys.
        Example:
             [ {'context': 'good morning!', 'response': 'hello what can i help you with today'},
             {'context': 'i'd like to book a table with italian food', 'response': 'i'm on it'}]
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
