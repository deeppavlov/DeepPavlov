from deeppavlov.models.go_bot.nlg.dto.nlg_response_interface import NLGObjectResponseInterface


class JSONNLGResponse(NLGObjectResponseInterface):
    def __init__(self, slot_values: dict, actions_tuple: tuple):
        self.slot_values = slot_values
        self.actions_tuple = actions_tuple

    def to_serializable_dict(self) -> dict:
        return {'+'.join(self.actions_tuple): self.slot_values}
