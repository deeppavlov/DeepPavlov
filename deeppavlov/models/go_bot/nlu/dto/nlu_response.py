from typing import Any, Dict, Tuple, List, Union, Optional

from deeppavlov.models.go_bot.nlu.dto.nlu_response_interface import NLUResponseInterface
from deeppavlov.models.go_bot.nlu.dto.text_vectorization_response import TextVectorizationResponse


class NLUResponse(NLUResponseInterface):
    """
    Stores the go-bot NLU knowledge: extracted slots and intents info, embedding and bow vectors.
    """
    def __init__(self, slots, intents, tokens):
        self.slots: Union[List[Tuple[str, Any]], Dict[str, Any]] = slots
        self.intents = intents
        self.tokens = tokens
        self.tokens_vectorized: Optional[TextVectorizationResponse] = None

    def set_tokens_vectorized(self, tokens_vectorized):
        self.tokens_vectorized = tokens_vectorized
