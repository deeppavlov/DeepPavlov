from typing import List

from deeppavlov.models.inferable import Inferable


class Agent(Inferable):
    def __init__(self, skills: List, commutator: Inferable):
        self.skills = skills
        self.commutator = commutator
        self.history = None

    def infer(self):
        pass
