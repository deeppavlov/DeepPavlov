from typing import Any, Dict, Tuple, List, Union

from deeppavlov.models.go_bot.nlu.dto.nlu_response_interface import NLUResponseInterface


class NLUResponse(NLUResponseInterface):
    def __init__(self, slots, intents, tokens):
        self.slots: Union[List[Tuple[str, Any]], Dict[str, Any]] = slots
        self.intents = intents
        self.tokens = tokens