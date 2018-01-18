import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.inferable import Inferable


@register('bow')
class BoW_encoder(Inferable):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _encode(self, utterance, vocab):
        bow = np.zeros([len(vocab)], dtype=np.int32)
        for word in utterance.split(' '):
            if word in vocab:
                idx = vocab[word]
                bow[idx] += 1
        return bow

    def infer(self, utterance, vocab):
        return self._encode(utterance, vocab)
