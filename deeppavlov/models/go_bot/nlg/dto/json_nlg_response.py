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

class VerboseJSONNLGResponse(JSONNLGResponse):

    @staticmethod
    def from_json_nlg_response(json_nlg_response: JSONNLGResponse) -> "VerboseJSONNLGResponse":
        verbose_json_nlg_response = VerboseJSONNLGResponse(json_nlg_response.slot_values,
                                                           json_nlg_response.actions_tuple)
        return verbose_json_nlg_response

    def get_nlu_info(self):
        intent_name = "start" if self.actions_tuple[0] == "start" else self.actions_tuple[0][len("utter_"):].split('{')[0]
        return {"intent": intent_name}
