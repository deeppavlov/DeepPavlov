from typing import Container
from deeppavlov.models.go_bot.nlg.dto.nlg_response_interface import NLGResponseInterface


class BatchNLGResponse:
    def __init__(self, nlg_responses: Container[NLGResponseInterface]):
        self.responses: Container[NLGResponseInterface] = nlg_responses
