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
    def __init__(self, save_path, load_path, level='token',
                 special_tokens=tuple(), default_token=None,
                 tokenizer=None, min_freq=0, **kwargs):

        super().__init__(load_path=load_path,
                         save_path=save_path,
                         **kwargs)

        self.special_tokens = special_tokens
        self.default_token = default_token
        self.min_freq = min_freq
        self.preprocess_fn = self._build_preprocess_fn(level, tokenizer)

        # TODO check via decorator
        self.reset()
        if self.load_path:
            self.load()

    @staticmethod
    def _build_preprocess_fn(level, tokenizer=None):
        def iter_level(utter):
            if isinstance(utter, list) and utter and isinstance(utter[0], dict):
                tokens = (u['text'] for u in utter)
            elif isinstance(utter, dict):
                tokens = [utter['text']]
            elif isinstance(utter, list) and (not utter or isinstance(utter[0], str) or isinstance(utter[0], tuple)):
                tokens = utter
            else:
                tokens = [utter]

            if tokenizer is not None:
                tokens = tokenizer([' '.join(tokens)])[0]
            tokens = filter(None, tokens)

            if level == 'token':
                yield from tokens
            elif level == 'char':
                for token in tokens:
                    yield from token
            else:
                raise ValueError("level argument is either equal to `token`"
                                 " or to `char`")

        def preprocess_fn(data):
            for d in data:
                yield from iter_level(d)

        return preprocess_fn

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._i2t[key]
        elif isinstance(key, str):
            return self._t2i[key]
        else:
            raise NotImplementedError("not implemented for type `{}`".format(type(key)))

    def __contains__(self, item):
        return item in self._t2i

    def __len__(self):
        return len(self._t2i)

    def keys(self):
        return (k for k, v in self.freqs.most_common() if k in self._t2i)

    def values(self):
        return (v for k, v in self.freqs.most_common() if k in self._t2i)

    def items(self):
        return ((k, v) for k, v in self.freqs.most_common() if k in self._t2i)

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

    def fit(self, *args):
        self.reset()
        self._train(
            tokens=filter(None, itertools.chain.from_iterable(
                map(self.preprocess_fn, zip(*args)))),
            counts=None,
            update=True
        )

    def _train(self, tokens, counts=None, update=True):
        counts = counts or itertools.repeat(1)
        if not update:
            self.reset()

        for token, cnt in zip(tokens, counts):
            self.freqs[token] += cnt

        index = len(self._t2i)
        for token, count in self.freqs.items():
            if token not in self._t2i and count >= self.min_freq:
                self._t2i[token] = index
                self._i2t[index] = token
                index += 1
        return

    def __call__(self, samples, **kwargs):
        return [self[s] for s in samples]

    def save(self):
        log.info("[saving vocabulary to {}]".format(self.save_path))

        with self.save_path.open('wt', encoding="utf8") as f:
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
                for ln in self.load_path.open('r', encoding="utf8"):
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
            # if not filter_paddings or idx != self.tok2idx('<PAD>'):
            toks.append(self._i2t[idx])
        return toks

    def tok2idx(self, tok):
        return self._t2i[tok]

    def toks2idxs(self, toks):
        return [self._t2i[tok] for tok in toks]

    def batch_toks2batch_idxs(self, b_toks):
        max_len = max(len(toks) for toks in b_toks)
        # Create array filled with paddings
        # batch = np.ones([len(b_toks), max_len]) * self.tok2idx('<PAD>')
        batch = np.zeros([len(b_toks), max_len])
        for n, tokens in enumerate(b_toks):
            idxs = self.toks2idxs(tokens)
            batch[n, :len(idxs)] = idxs
        return batch

    def batch_idxs2batch_toks(self, b_idxs, filter_paddings=False):
        return [self.idxs2toks(idxs, filter_paddings) for idxs in b_idxs]

