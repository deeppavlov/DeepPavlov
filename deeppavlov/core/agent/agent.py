from typing import List, Dict

from deeppavlov.core.models.component import Component


# TODO Create this class dynamically?
class Agent(Component):
    def __init__(self, skill_configs: List[Dict], commutator_config: Dict, *args, **kwargs):
        self.skill_configs = skill_configs
        self.commutator_config = commutator_config
        self.history = []
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        pass
