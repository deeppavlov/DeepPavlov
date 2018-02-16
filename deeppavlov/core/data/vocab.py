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

from collections import Counter, defaultdict
import itertools
from pathlib import Path

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator

log = get_logger(__name__)


@register('default_vocab')
class DefaultVocabulary(Estimator):
    def __init__(self, inputs, save_path, load_path, level='token',
                 special_tokens=tuple(), default_token=None,
                 tokenize=False, train_now=False, *args, **kwargs):

        super().__init__(load_path=load_path,
                         save_path=save_path,
                         train_now=train_now,
                         mode=kwargs['mode'])

        self.special_tokens = special_tokens
        self.default_token = default_token
        self.preprocess_fn = self._build_preprocess_fn(inputs, level, tokenize)

        # TODO check via decorator
        self.reset()
        self.load()

    @staticmethod
    def _build_preprocess_fn(inputs, level, tokenize):
        def iter_level(utter):
            if isinstance(utter, list) and isinstance(utter[0], dict):
                utter = ' '.join(u['text'] for u in utter)
            elif isinstance(utter, dict):
                utter = utter['text']

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
    def fit(self, x, y):
        self.reset()
        self._train(
            tokens=filter(None, itertools.chain.from_iterable(
                map(self.preprocess_fn, zip(x, y)))),
            counts=None,
            update=True
        )

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

    def __call__(self, samples, **kwargs):
        return [self[s] for s in samples]

    def save(self):
        log.info("[saving vocabulary to {}]".format(self.save_path))

        with self.save_path.open('wt') as f:
            for n in range(len(self._t2i)):
                token = self._i2t[n]
                cnt = self.freqs[token]
                f.write('{}\t{:d}\n'.format(token, cnt))

    # @check_path_exists()
    def load(self):
        if self.load_path:
            if self.load_path.is_file():
                log.info("[loading vocabulary from {}]".format(self.load_path))
                tokens, counts = [], []
                for ln in self.load_path.open('r'):
                    token, cnt = ln.split('\t', 1)
                    tokens.append(token)
                    counts.append(int(cnt))
                self._train(tokens=tokens, counts=counts, update=True)
            elif isinstance(self.load_path, Path):
                if not self.load_path.parent.is_dir():
                    raise ConfigError("Provided `load_path` for {} doesn't exist!".format(
                        self.__class__.__name__))
        else:
            raise ConfigError("`load_path` for {} is not provided!".format(self))

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
