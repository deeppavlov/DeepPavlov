from collections import Counter, defaultdict
import numpy as np
from deeppavlov.core.common.registry import register_model
from deeppavlov.core.models.inferable import Inferable
from overrides import overrides


@register_model('default_vocab')
class Vocabulary(Inferable):
    def __init__(self, tokens=None, special_tokens=tuple(), dict_file_path=None):
        if dict_file_path is not None:
            if tokens is None:
                Warning('Tokens provided along with external dictionary file!')
            tokens = self.load(dict_file_path)
        self._t2i = dict()
        # We set default ind to position of <UNK> in SPECIAL_TOKENS
        # because the tokens will be added to dict in the same order as
        # in special_tokens
        default_ind = 0
        self._t2i = defaultdict(lambda: default_ind)
        self._i2t = dict()
        self.frequencies = Counter()

        self.counter = 0
        for token in special_tokens:
            self._t2i[token] = self.counter
            self.frequencies[token] += 0
            self._i2t[self.counter] = token
            self.counter += 1
        if tokens is not None:
            self.update_dict(tokens)

    def update_dict(self, tokens):
        for token in tokens:
            if token not in self._t2i:
                self._t2i[token] = self.counter
                self._i2t[self.counter] = token
                self.counter += 1
            self.frequencies[token] += 1

    def idx2tok(self, idx):
        return self._i2t[idx]

    def idxs2toks(self, idxs, filter_paddings=False):
        toks = []
        for idx in idxs:
            if not filter_paddings or idx != self.tok2idx('<PAD>'):
                toks.append(self._i2t[idx])
        return toks

    @overrides
    def infer(self, samples):
        if not isinstance(samples, str):
            return [self.infer(sample) for sample in samples]
        else:
            return self.tok2idx(samples)

    def tok2idx(self, tok):
        return self._t2i[tok]

    def toks2idxs(self, toks):
        return [self._t2i[tok] for tok in toks]

    def batch_toks2batch_idxs(self, b_toks):
        max_len = max(len(toks) for toks in b_toks)
        # Create array filled with paddings
        batch = np.ones([len(b_toks), max_len]) * self.tok2idx('<PAD>')
        for n, tokens in enumerate(b_toks):
            idxs = self.toks2idxs(tokens)
            batch[n, :len(idxs)] = idxs
        return batch

    def batch_idxs2batch_toks(self, b_idxs, filter_paddings=False):
        return [self.idxs2toks(idxs, filter_paddings) for idxs in b_idxs]

    def is_pad(self, x_t):
        assert type(x_t) == np.ndarray
        return x_t == self.tok2idx('<PAD>')

    def __getitem__(self, key):
        return self._t2i[key]

    def __len__(self):
        return self.counter

    def __contains__(self, item):
        return item in self._t2i

    def load(self, dict_file_path):
        tokens = list()
        with open(dict_file_path) as f:
            for line in f:
                if len(line) > 0:
                    tokens.append(line)
        return tokens

