from typing import List
from numpy.random import choice

from deeppavlov.models.inferable import Inferable
from deeppavlov.common.registry import register_model

@register_model("random")
class RandomCommutator(Inferable):
    def __init__(self):
        pass

    def _commutate(self, predictions, history):
        # commutating
        winner = choice(predictions)
        return winner

    def infer(self, predictions, history):
        return self._commutate(predictions,  history)
