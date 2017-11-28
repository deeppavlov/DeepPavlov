from typing import Dict

import numpy as np

from deeppavlov.common.registry import register_model
from deeppavlov.models.model import Model


@register_model('bow')
class BoW_encoder(Model):
    def __init__(self):
        pass

    def _encode(self, utterance, vocab):
        bow = np.zeros([len(vocab)], dtype=np.int32)
        for word in utterance.split(' '):
            if word in vocab:
                idx = vocab.index(word)
                bow[idx] += 1
        return bow

    def infer(self, utterance, vocab):
        return self._encode(utterance, vocab)
