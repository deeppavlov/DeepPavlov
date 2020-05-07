from deeppavlov.models.go_bot.nlg.dto.nlg_response_interface import NLGResponseInterface


class TemplatedNLGResponse(NLGResponseInterface):
    def __init__(self, templated_response: str):
        self.response_text = templated_response
