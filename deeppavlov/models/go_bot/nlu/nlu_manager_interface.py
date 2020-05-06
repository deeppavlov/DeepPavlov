from abc import ABCMeta, abstractmethod


class NLUManagerInterface(metaclass=ABCMeta):
    @abstractmethod
    def nlu(self, text):
        pass

    @abstractmethod
    def num_of_known_intents(self):
        """:returns: the number of intents known to the NLU module"""
        pass