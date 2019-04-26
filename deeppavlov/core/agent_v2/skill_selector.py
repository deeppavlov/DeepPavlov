from typing import Dict, List
from collections import defaultdict

from deeppavlov.core.agent_v2.config import SKILLS
from deeppavlov.core.agent_v2.service import Service


class ChitchatQASelector(Service):
    SKILL_NAMES_MAP = {
        "chitchat": ["chitchat", "hellobot", "sberchat", "gen_chitchat"],
        "odqa": ["odqa", "kbqa", "generalqa", "mailruqa"]
    }

    def __init__(self, rest_caller):
        super().__init__(rest_caller)
        available_names = [s['name'] for s in SKILLS]
        self.skill_names_map = defaultdict(list)
        for selector_names, agent_names in self.SKILL_NAMES_MAP.items():
            names = {an for an in agent_names if an in available_names}
            self.skill_names_map[selector_names] += list(names)

    def __call__(self, state: Dict) -> List[List[str]]:
        """
        Select a skill.
        Args:
            state:

        Returns: a list of skill names for each utterance

        """
        response = self.rest_caller(payload=state)
        # TODO refactor riseapi so it would not return keys from dp config?
        predicted_names = [el[self.rest_caller.names[0]]['skill_names'] for el in response]
        skill_names = [self.skill_names_map[name] for name in predicted_names]
        return skill_names
