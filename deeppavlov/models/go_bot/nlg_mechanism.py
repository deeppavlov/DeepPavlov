import re

from deeppavlov.core.commands.utils import expand_path
import deeppavlov.models.go_bot.templates as go_bot_templates


class NLGHandler:
    def __init__(self, template_path, template_type, api_call_action):
        template_path = expand_path(template_path)
        template_type = getattr(go_bot_templates, template_type)
        self.templates = go_bot_templates.Templates(template_type).load(template_path)

        self.api_call_id = -1
        if api_call_action is not None:
            self.api_call_id = self.templates.actions.index(api_call_action)

    def encode_response(self, act: str) -> int:
        return self.templates.actions.index(act)

    def decode_response(self, action_id: int, tracker_slotfilled_state: dict) -> str:
        """
        Convert action template id and entities from tracker to final response.
        """
        resp = self.generate_slotfilled_text_for_action(action_id, tracker_slotfilled_state)
        # in api calls replace unknown slots to "dontcare"
        if action_id == self.api_call_id:
            resp = re.sub("#([A-Za-z]+)", "dontcare", resp).lower()
        return resp

    def generate_slotfilled_text_for_action(self, action_id: int, slots: dict):
        """
        Generate text for the predicted speech action using the pattern provided for the action.
        The slotfilled state provides info to encapsulate to the pattern.
        """
        text = self.templates.templates[action_id].generate_text(slots)
        return text

    def num_of_known_actions(self) -> int:
        return len(self.templates)