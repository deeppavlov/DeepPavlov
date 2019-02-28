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

        # TODO put this functionality to configs.dp_assistant.sselector_chitchat_odqa skill,
        # TODO so it would return only classes names
        skill_names = []
        for el in response:
            probas = el['chitchat_odqa']['y_pred_probas']
            if probas[0] > probas[1]:
                skill_names.append('chitchat')
            else:
                skill_names.append('odqa')
        return skill_names
