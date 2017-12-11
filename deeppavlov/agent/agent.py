from typing import List, Dict

from deeppavlov.models.inferable import Inferable

# Create this class dynamically
class Agent(Inferable):
    def __init__(self, skill_configs: List[Dict], commutator_config: Inferable):
        self.skill_configs = skill_configs
        self.commutator_config = commutator_config
        self.history = []

    def infer(self):
        pass
