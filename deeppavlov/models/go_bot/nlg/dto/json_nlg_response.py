from deeppavlov.models.go_bot.nlg.dto.nlg_response_interface import NLGObjectResponseInterface


class JSONNLGResponse(NLGObjectResponseInterface):
    """
    The NLG output unit that stores slot values and predicted actions info.
    """
    def __init__(self, slot_values: dict, actions_tuple: tuple):
        self.slot_values = slot_values
        self.actions_tuple = actions_tuple

    def to_serializable_dict(self) -> dict:
        return {'+'.join(self.actions_tuple): self.slot_values}
