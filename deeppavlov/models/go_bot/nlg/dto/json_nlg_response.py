import json

from deeppavlov.models.go_bot.nlg.dto.nlg_response_interface import NLGResponseInterface


class JSONNLGResponse(NLGResponseInterface):
    def __init__(self, response_dict: dict):
        self.json = json.dumps(response_dict)

