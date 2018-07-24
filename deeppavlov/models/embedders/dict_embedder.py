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

import numpy as np
from pathlib import Path
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.serializable import Serializable
from typing import List

log = get_logger(__name__)


@register('dict_emb')
class DictEmbedder(Component, Serializable):
    def __init__(self, load_path, save_path=None, dim=100, **kwargs):
        super().__init__(save_path=save_path, load_path=load_path)
        self.tok2emb = {}
        self.dim = dim

        self.load()

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self):
        """
        Load dictionary of embeddings from file.
        """

        if not Path(self.load_path).exists():
            raise FileNotFoundError(
                'There is no dictionary of embeddings <<{}>> file provided.'.format(
                    self.load_path))
        else:
            log.info('Loading existing dictionary of embeddings from {}'.format(self.load_path))

            with open(self.load_path, encoding='utf8') as fin:
                for line in fin:
                    values = line.rsplit(sep=' ', maxsplit=self.dim)
                    assert (len(values) == self.dim + 1)
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.tok2emb[word] = coefs

    @overrides
    def __call__(self, batch, mean=False, *args, **kwargs):
        """
        Embed data
        """
        return [self._encode(sentence, mean) for sentence in batch]

    def _encode(self, sentence: str, mean):
        tokens = sentence.split()
        embedded_tokens = []
        for t in tokens:
            try:
                emb = self.tok2emb[t]
            except KeyError:
                emb = np.zeros(self.dim, dtype=np.float32)
                self.tok2emb[t] = emb
            embedded_tokens.append(emb)

        if mean:
            filtered = [et for et in embedded_tokens if np.any(et)]
            if filtered:
                return np.mean(filtered, axis=0)
            return np.zeros(self.dim, dtype=np.float32)

        return embedded_tokens
