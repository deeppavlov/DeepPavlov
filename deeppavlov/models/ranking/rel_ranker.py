from typing import List, Tuple, Union

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.embedders.abstract_embedder import Embedder


@register('two_sentences_emb')
class TwoSentencesEmbedder(Component):
    """This class is used for embedding of two sentences."""

    def __init__(self, embedder: Embedder, **kwargs):
        """

        Args:
            embedder: what embedder to use: Glove, Fasttext or other
            **kwargs:
        """
        self.embedder = embedder

    def __call__(self, sentence_tokens_1: List[List[str]], sentence_tokens_2: List[List[str]]) -> \
            Tuple[List[Union[list, np.ndarray]], List[Union[list, np.ndarray]]]:
        sentence_token_embs_1 = self.embedder(sentence_tokens_1)
        sentence_token_embs_2 = self.embedder(sentence_tokens_2)
        return sentence_token_embs_1, sentence_token_embs_2
