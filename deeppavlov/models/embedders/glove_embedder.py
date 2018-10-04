# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from overrides import overrides
from typing import List, Union, Generator
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.data.utils import zero_pad
from typing import List

log = get_logger(__name__)


@register('glove')
class GloVeEmbedder(Component, Serializable):
    """
    Class implements GloVe embedding model

    Args:
        load_path: path where to load pre-trained embedding model from
        save_path: is not used because model is not trainable; therefore, it is unchangable
        dim: dimensionality of fastText model
        pad_zero: whether to pad samples or not
        mean: whether to return mean token embedding
        **kwargs: additional arguments

    Attributes:
        model: GloVe model instance
        tok2emb: dictionary with already embedded tokens
        dim: dimension of embeddings
        pad_zero: whether to pad sequence of tokens with zeros or not
        load_path: path with pre-trained GloVe model
        mean: whether to return mean token embedding
    """
    def __init__(self, load_path: [str, Path], save_path: [str, Path] = None, dim: int = 100, pad_zero: bool = False,
                 mean: bool = False, **kwargs) -> None:
        """
        Initialize embedder with given parameters
        """
        super().__init__(save_path=save_path, load_path=load_path)
        self.tok2emb = {}
        self.dim = dim
        self.pad_zero = pad_zero
        self.model = self.load()

    def save(self, *args, **kwargs) -> None:
        """
        Class do not save loaded model again as it is not trained during usage

        Args:
            *args: arguments
            **kwargs: arguments

        Returns:
            None
        """
        raise NotImplementedError

    def load(self, *args, **kwargs) -> KeyedVectors:
        """
        Load dict of embeddings from given file

        Args:
            *args: arguments
            **kwargs: arguments

        Returns:

        """
        # Check that header with n_words emb_dim present
        with open(self.load_path, encoding='utf8') as f:
            header = f.readline()
            if len(header.split()) != 2:
                raise RuntimeError('The GloVe file must start with number_of_words embeddings_dim line! '
                                   'For example "40000 100" for 40000 words vocabulary and 100 embeddings '
                                   'dimension.')

        if self.load_path and self.load_path.is_file():
            log.info("[loading embeddings from `{}`]".format(self.load_path))
            model_file = str(self.load_path)
            model = KeyedVectors.load_word2vec_format(model_file)
        else:
            log.error('No pretrained GloVe model provided or provided load_path "{}" is incorrect.'
                      .format(self.load_path))
            sys.exit(1)

        return model

    def __iter__(self) -> Generator:
        """
        Iterate over all words from fastText model vocabulary

        Returns:
            iterator
        """
        yield from self.model.vocab

    @overrides
    def __call__(self, batch: List[List[str]], *args, **kwargs) -> List[Union[list, np.ndarray]]:
        """
        Embed sentences from batch

        Args:
            batch: list of tokenized text samples
            mean: whether to return mean embedding of tokens per sample
            *args: arguments
            **kwargs: arguments

        Returns:
            embedded batch
        """
        embedded = []
        for n, sample in enumerate(batch):
            embedded.append(self._encode(sample))
        if self.pad_zero:
            embedded = zero_pad(embedded)
        return embedded

    def _encode(self, tokens: List[str]) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed one text sample

        Args:
            tokens: tokenized text sample
            mean: whether to return mean embedding of tokens per sample

        Returns:
            list of embedded tokens or array of mean values
        """
        embedded_tokens = []
        for t in tokens:
            try:
                emb = self.tok2emb[t]
            except KeyError:
                try:
                    emb = self.model[t][:self.dim]
                except KeyError:
                    emb = np.zeros(self.dim, dtype=np.float32)
                self.tok2emb[t] = emb
            embedded_tokens.append(emb)

        if self.mean:
            filtered = [et for et in embedded_tokens if np.any(et)]
            if filtered:
                return np.mean(filtered, axis=0)
            return np.zeros(self.dim, dtype=np.float32)

        return embedded_tokens
