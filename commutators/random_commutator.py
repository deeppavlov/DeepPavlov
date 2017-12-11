from typing import List
from random import choice

from deeppavlov.models.inferable import Inferable
from deeppavlov.common.registry import register_model

@register_model("random")
class RandomCommutator(Inferable):
    def __init__(self):
        pass

    def _commutate(self, predictions, history):
        idx = choice(range(len(predictions)))
        winner = predictions[idx]
        name = list(winner.keys())[0]
        prediction = list(winner.values())[0]
        return idx, name, prediction

    def infer(self, predictions, history):
        return self._commutate(predictions,  history)
