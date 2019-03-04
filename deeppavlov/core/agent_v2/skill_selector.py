from typing import Dict, List, Tuple


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


class ChitchatOdqaSelector(SkillSelector):
    SKILL_NAMES_MAP = {
        "chitchat": "chitchat",
        "odqa": "odqa"
    }

    def __init__(self, rest_caller):
        super().__init__(rest_caller)

    def __call__(self, state: Dict):
        """
        Select a skill.
        Args:
            state:

        Returns: a skill name

        """
        response = self.rest_caller(payload=state)
        # TODO refactor riseapi so it would not return keys?
        skill_names = [el[self.rest_caller.names[0]]['skill_names'] for el in response]
        return skill_names
