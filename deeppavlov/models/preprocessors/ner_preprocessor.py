import errno
import os
from logging import getLogger
from typing import List

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator

log = getLogger(__name__)


@register("ner_vocab")
class NerVocab(Estimator):
    """ Implementation of the NER vocabulary

    Params:
        word_file_path: the path to the pre-trained word embedding model
        save_path: the folder path to save dictionary files
        load_path: the folder path from which the dictionary files are loaded
        char_level: the flag arg indicating the character vocabulary
    """

    def __init__(self,
                 word_file_path=None,
                 save_path=None,
                 load_path=None,
                 char_level=False,
                 **kwargs):

        super().__init__(save_path=save_path, load_path=load_path, **kwargs)

        self.word_file_path = word_file_path
        self.char_level = char_level

        if word_file_path is not None:
            self.load_from_file(word_file_path)
            if self.save_path is not None:
                self.save_to_file(self.save_path)
        elif self.load_path is not None:
            self.load_from_file(self.load_path)

    def load_from_file(self, filename):
        if filename is None or not os.path.exists(filename):
            return

        self._t2i, self._i2t = {}, {}
        for i, line in enumerate(open(file=filename, mode="r", encoding="utf-8").readlines()):
            word = line.strip()
            self._t2i[word] = i
            self._i2t[i] = word

    def save_to_file(self, filename):
        if filename is None:
            return

        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(file=filename, mode="w", encoding="utf-8") as fo:
            for word in self._t2i.keys():
                fo.write("{}\n".format(word))

    def fit(self, sents: [List[List[str]]], *args):
        if self.word_file_path is not None:
            return

        if self.char_level:
            items = set([char for sent in sents for word in sent for char in word])
        else:
            items = set([word for sent in sents for word in sent])
        items = ["<UNK>", "<PAD>"] + list(items)

        self._t2i = {k: v for v, k in enumerate(items)}
        self._i2t = {k: v for k, v in enumerate(items)}

        self.save_to_file(self.save_path)

    def pad_batch(self, tokens: List[List[int]]):
        """ Create padded batch of words, tags, chunk pos, even batch of characters

        Params:
            tokens: list of raw words, pos, chunk, or tags.

        Returns:
            the padded batch
        """

        batch_size = len(tokens)

        if not self.char_level:
            max_len = max([len(seq) for seq in tokens])
            padded_batch = np.full((batch_size, max_len), self._t2i["<PAD>"])
            for i, seq in enumerate(tokens):
                padded_batch[i, :len(seq)] = seq
        else:
            max_len_seq = max([len(seq) for seq in tokens])
            if max_len_seq == 0:
                max_len_sub_seq = 0
            else:
                max_len_sub_seq = max([len(sub_seq) for seq in tokens for sub_seq in seq])
            padded_batch = np.full((batch_size, max_len_seq, max_len_sub_seq), self._t2i["<PAD>"])
            for i, seq in enumerate(tokens):
                for j, sub_seq in enumerate(seq):
                    padded_batch[i, j, :len(sub_seq)] = sub_seq
        return padded_batch

    def __call__(self, sents, **kwargs):
        if not self.char_level:
            sents_ind = [[self._t2i[word] if word in self._t2i else 0 for word in sent] for sent in sents]
        else:
            sents_ind = [[[self._t2i[char] if char in self._t2i else 0 for char in word] for word in sent] for sent in
                         sents]
        padded_sents = self.pad_batch(sents_ind)

        return padded_sents

    def load(self, *args, **kwargs):
        log.info("[loading vocabulary from {}]".format(self.load_path))
        if self.load_path is not None:
            self.load_from_file(self.load_path)

    def save(self, *args, **kwargs):
        log.info("[saving vocabulary to {}]".format(self.save_path))
        if not os.path.exists(os.path.dirname(self.save_path)):
            try:
                os.makedirs(os.path.dirname(self.save_path))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        self.save_to_file(self.save_path)

    @property
    def len(self):
        return len(self._t2i)

    @property
    def t2i(self):
        return self._t2i

    @property
    def i2t(self):
        return self._i2t
