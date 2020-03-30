import re
from logging import getLogger
from typing import List

import numpy as np

# from deeppavlov.models.go_bot.network import log
import deeppavlov.models.go_bot.templates as templ
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.models.go_bot.tracker import DialogueStateTracker
from deeppavlov.models.go_bot.utils import GobotAttnHyperParams

log = getLogger(__name__)


class DataHandler:

    def __init__(self, debug, template_path, template_type, word_vocab, bow_embedder, api_call_action, embedder):
        self.debug = debug

        template_path = expand_path(template_path)
        template_type = getattr(templ, template_type)
        log.info(f"[loading templates from {template_path}]")
        self.templates = templ.Templates(template_type).load(template_path)  # upper-level model logic
        log.info(f"{len(self.templates)} templates loaded.")

        self.api_call_id = -1  # api call should have smth like action index
        if api_call_action is not None:
            self.api_call_id = self.templates.actions.index(api_call_action)  # upper-level model logic

        self.word_vocab = word_vocab
        self.bow_embedder = bow_embedder
        self.embedder = embedder

    def use_bow_embedder(self):
        return callable(self.bow_embedder)

    def word_vocab_size(self):
        return len(self.word_vocab) if self.word_vocab else None


    def encode_response(self, act: str) -> int:
        # conversion
        return self.templates.actions.index(act)

    def decode_response(self, action_id: int, tracker: DialogueStateTracker) -> str:
        """
        Convert action template id and entities from tracker
        to final response.
        """
        # conversion
        template = self.templates.templates[int(action_id)]

        slots = tracker.get_state()
        if tracker.db_result is not None:
            for k, v in tracker.db_result.items():
                slots[k] = str(v)

        resp = template.generate_text(slots)
        # in api calls replace unknown slots to "dontcare"
        if action_id == self.api_call_id:
            # todo: move api_call_id here
            resp = re.sub("#([A-Za-z]+)", "dontcare", resp).lower()
        return resp

    def encode_tokens(self, tokens: List[str], mean_embeddings):

        bow_features = self.bow_encode_tokens(tokens)
        tokens_exexe = self.embed_tokens(tokens, mean_embeddings)

        return bow_features, tokens_exexe

    def embed_tokens(self, tokens, mean_embeddings):
        tokens_embedded = None  # todo worst name ever
        if callable(self.embedder):
            tokens_embedded = self.embedder([tokens], mean=mean_embeddings)[0]
        return tokens_embedded

    def bow_encode_tokens(self, tokens):
        bow_features = []
        if self.use_bow_embedder():
            tokens_idx = self.word_vocab(tokens)
            bow_features = self.bow_embedder([tokens_idx])[0]
            bow_features = bow_features.astype(np.float32)
        return bow_features
