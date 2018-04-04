from deeppavlov.core.models.component import Component
import numpy as np


class CapitalizationPreprocessor(Component):
    """ Patterns:
        - no capitals
        - single capital single character
        - single capital multiple characters
        - all capitals multiple characters
    """
    def __init__(self, pad_zeros=True):
        self.pad_zeros = pad_zeros
        self.n_features = 4

    @property
    def n_features(self):
        return self.n_features

    def __call__(self, tokens_batch, **kwargs):
        cap_batch = []
        batch_size = len(tokens_batch)
        max_batch_len = 0
        for utterance in tokens_batch:
            cap_list = []
            max_batch_len = max(max_batch_len, len(utterance))
            for token in utterance:
                cap = np.zeros(4, np.float32)
                # Check the case and produce corresponding one-hot
                if token[0].islower():
                    cap[0] = 1
                elif len(token) == 1 and token[0].isupper():
                    cap[1] = 1
                elif len(token) > 1 and token[0].isupper():
                    cap[2] = 1
                elif all(ch.isupper() for ch in token):
                    cap[3] = 1
                else:
                    raise RuntimeError('Unknown capitalization of token {}!'.format(token))
                cap_list.append(cap)
            cap_batch.append(cap)
        if self.pad_zeros:
            cap_batch_padded = np.zeros([batch_size, max_batch_len, self.n_features], np.float32)
            for n, utterance_cap in cap_batch:
                for k, token_cap in utterance_cap:
                    cap_batch_padded[n, k] = token_cap
            return cap_batch_padded
        else:
            return cap_batch