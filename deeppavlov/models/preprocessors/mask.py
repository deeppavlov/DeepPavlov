from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
import numpy as np


@register('mask')
class Mask(Component):
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __call__(tokens_batch, **kwargs):
        """Takes batch of tokens and returns the lemmatized tokens"""
        batch_size = len(tokens_batch)
        max_len = max(len(utt) for utt in tokens_batch)
        mask = np.zeros([batch_size, max_len], dtype=np.float32)
        for n, utterance in enumerate(tokens_batch):
            mask[n, :len(utterance)] = 1

        return mask
