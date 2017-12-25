"""
Here is an abstract class for storing different vocabularities (words, symbols,
etc.)
"""

from abc import abstractmethod

from deeppavlov.core.models.inferable import Inferable


class Vocabulary(Inferable):

    @abstractmethod
    def update_dict(self, tokens):
        pass

    @abstractmethod
    def iter_all(self):
        pass

    @abstractmethod
    def save(self, fname):
        pass
