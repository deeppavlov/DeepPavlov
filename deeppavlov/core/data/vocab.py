from collections import Counter, defaultdict
import itertools
import numpy as np
import os

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.attributes import check_path_exists, check_attr_true


@register('default_vocab')
class DefaultVocabulary(Trainable, Inferable):
    def __init__(self, inputs, level='token', ser_path=None,
                 ser_dir='vocabs', ser_file='vocab.txt',
                 special_tokens=tuple(), default_token=None,
                 tokenize=False, train_now=False, *args, **kwargs):

        super().__init__(ser_path=ser_path,
                         ser_dir=ser_dir,
                         ser_file=ser_file,
                         train_now=train_now,
                         mode=kwargs['mode'])

        self.special_tokens = special_tokens
        self.default_token = default_token
        self.preprocess_fn = self._build_preprocess_fn(inputs, level, tokenize)

        # TODO check via decorator
        self.reset()

        if not self.ser_path.is_file():
            with self.ser_path.open('a'):
                os.utime(self.ser_path, None)

        self.load()

    @staticmethod
    def _build_preprocess_fn(inputs, level, tokenize):
        def iter_level(utter):
            if tokenize:
                utter = utter.split()
            if level == 'token':
                yield from utter
            elif level == 'char':
                for token in utter:
                    yield from token
            else:
                raise ValueError("level argument is either equal to `token`"
                                 " or to `char`")

        def preprocess_fn(data):
            for f in inputs:
                if f == 'x':
                    yield from iter_level(data[0])
                elif f == 'y':
                    yield from iter_level(data[1])
                else:
                    yield from iter_level(data[2][f])

        return preprocess_fn

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._i2t[key]
        elif isinstance(key, str):
            return self._t2i[key]
        else:
            return NotImplemented("not implemented for type `{}`".format(type(key)))

    def __contains__(self, item):
        return item in self._t2i

    def __len__(self):
        return len(self.freqs)

    def keys(self):
        return (k for k, v in self.freqs.most_common())

    def values(self):
        return (v for k, v in self.freqs.most_common())

    def items(self):
        return self.freqs.most_common()

    def reset(self):
        # default index is the position of default_token
        if self.default_token is not None:
            default_ind = self.special_tokens.index(self.default_token)
        else:
            default_ind = 0
        self._t2i = defaultdict(lambda: default_ind)
        self._i2t = dict()
        self.freqs = Counter()

        for i, token in enumerate(self.special_tokens):
            self._t2i[token] = i
            self._i2t[i] = token
            self.freqs[token] += 0

    @check_attr_true('train_now')
    def train(self, data):
        self.reset()
        self._train(
            tokens=filter(None, itertools.chain.from_iterable(
                map(self.preprocess_fn, data))),
            counts=None,
            update=True
        )
        self.save()

    def _train(self, tokens, counts=None, update=True):
        counts = counts or itertools.repeat(1)
        if not update:
            self.reset()

        index = len(self.freqs)
        for token, cnt in zip(tokens, counts):
            if token not in self._t2i:
                self._t2i[token] = index
                self._i2t[index] = token
                index += 1
            self.freqs[token] += cnt

    def infer(self, samples):
        return [self.__getitem__(s) for s in samples]

    def save(self):
        print("[saving vocabulary to `{}`]".format(self.ser_path))

        with self.ser_path.open('wt') as f:
            for n in range(len(self._t2i)):
                token = self._i2t[n]
                cnt = self.freqs[token]
                f.write('{}\t{:d}\n'.format(token, cnt))

    @check_path_exists()
    def load(self):
        print("[loading vocabulary from `{}`]".format(self.ser_path))
        tokens, counts = [], []
        for ln in self.ser_path.open('r'):
            token, cnt = ln.split('\t', 1)
            tokens.append(token)
            counts.append(int(cnt))
        self._train(tokens=tokens, counts=counts, update=True)

    def idx2tok(self, idx):
        return self._i2t[idx]

    def idxs2toks(self, idxs, filter_paddings=False):
        toks = []
        for idx in idxs:
            if not filter_paddings or idx != self.tok2idx('<PAD>'):
                toks.append(self._i2t[idx])
        return toks

    def iter_all(self):
        for token in self.frequencies:
            yield token

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
