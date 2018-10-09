# originally based on https://github.com/tensorflow/models/tree/master/lm_1b
import glob
import random

import numpy as np

from typing import List

from bilm.data import Batcher

class InferBatcher(Batcher):
    '''
    Batch sentences of tokenized text into character id matrices.
    '''
    def __init__(self, lm_vocab_file: str, max_token_length: int, **kwargs):
        super(InferBatcher, self).__init__(lm_vocab_file, max_token_length, **kwargs)

    def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_length = max(len(sentence) for sentence in sentences) + 2


        X_char_ids = np.zeros((n_sentences, max_length, self._max_token_length))

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            char_ids_without_mask = self._lm_vocab.encode_chars(
                sent, split=False)
            # add one so that 0 is the mask value
            X_char_ids[k, :length, :] = char_ids_without_mask

        return X_char_ids
