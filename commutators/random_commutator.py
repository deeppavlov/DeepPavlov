from typing import List
from numpy.random import choice

from deeppavlov.models.inferable import Inferable

class RandomCommutator(Inferable):
    def __init__(self):
        pass

    def commutate(self, inputs: List):
        # commutating
        winner = choice(inputs)
        return winner