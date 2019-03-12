from typing import Dict, List, Tuple
from collections import defaultdict

from deeppavlov.core.agent_v2.config import SKILLS


class SkillSelector:
    def __init__(self, rest_caller=None):
        self.rest_caller = rest_caller

    def __call__(self, state: Dict) -> Tuple[List[str], List[str], List[float]]:
        """
        Select a single response for each dialog in the state.

        Args:
            state:

        Returns: a list of skill names

        """
        raise NotImplementedError


class ChitchatQASelector(SkillSelector):
    SKILL_NAMES_MAP = {
        "chitchat": ["chitchat", "hellobot"],
        "odqa": ["odqa"]
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
