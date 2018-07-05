"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# originally based on https://github.com/allenai/bilm-tf/blob/master/bilm/data.py

from bilm import UnicodeCharsVocabulary, Batcher
import numpy as np


class ExtUnicodeCharsVocabulary(UnicodeCharsVocabulary):
    """
    Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.
    """
    def __init__(self, vocab, max_word_length, **kwargs):
        # rewritten __init__ of Vocabulary
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1
        self._bos = -1
        self._eos = -1

        idx = 0
        for word in vocab:
            word_name = word.strip()
            if word_name == '<S>':
                self._bos = idx
            elif word_name == '</S>':
                self._eos = idx
            elif word_name == '<UNK>':
                self._unk = idx
            if word_name == '!!!MAXTERMID':
                continue

            self._id_to_word.append(word_name)
            self._word_to_id[word_name] = idx
            idx += 1

        # check to has special tokens
        if self._bos == -1 or self._eos == -1 or self._unk == -1:
            raise ValueError("Ensure the vocabulary file has "
                             "<S>, </S>, <UNK> tokens")

        # rewritten __init__ of UnicodeCharsVocabulary
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260  # <padding>

        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length], dtype=np.int32)

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r
        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars
        # TODO: properly handle <UNK>

class ExtBatcher(Batcher):
    """
    Batch sentences of tokenized text into character id matrices.
    """
    def __init__(self, vocab: str, max_token_length: int):
        """
        vocab = vocabulary (list of unique tokens ordered by frequency)
        max_token_length = the maximum number of characters in each token
        """
        self._lm_vocab = ExtUnicodeCharsVocabulary(
            vocab, max_token_length
        )
        self._max_token_length = max_token_length
