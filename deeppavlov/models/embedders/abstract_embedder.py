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

from abc import ABCMeta, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import List, Union, Iterator

import numpy as np
from overrides import overrides

from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

log = getLogger(__name__)


class Embedder(Component, Serializable, metaclass=ABCMeta):
    """
    Class implements fastText embedding model

    Args:
        load_path: path where to load pre-trained embedding model from
        pad_zero: whether to pad samples or not

    Attributes:
        model: model instance
        tok2emb: dictionary with already embedded tokens
        dim: dimension of embeddings
        pad_zero: whether to pad sequence of tokens with zeros or not
        mean: whether to return one mean embedding vector per sample
        load_path: path with pre-trained fastText binary model
    """

    def __init__(self, load_path: Union[str, Path], pad_zero: bool = False, mean: bool = False, **kwargs) -> None:
        """
        Initialize embedder with given parameters
        """
        super().__init__(save_path=None, load_path=load_path)
        self.tok2emb = {}
        self.pad_zero = pad_zero
        self.mean = mean
        self.dim = None
        self.model = None
        self.load()

    @overrides
    def save(self) -> None:
        """
        Class does not save loaded model again as it is not trained during usage
        """
        raise NotImplementedError

    @overrides
    def __call__(self, batch: List[List[str]], mean: bool = None) -> List[Union[list, np.ndarray]]:
        """
        Embed sentences from batch

        Args:
            batch: list of tokenized text samples
            mean: whether to return mean embedding of tokens per sample

        Returns:
            embedded batch
        """
        batch = [self._encode(sample, mean) for sample in batch]
        if self.pad_zero:
            batch = zero_pad(batch)
        return batch

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        """
        Iterate over all words from the model vocabulary

        Returns:
            iterator
        """

    @abstractmethod
    def _get_word_vector(self, w: str) -> np.ndarray:
        """
        Embed a word using ``self.model``

        Args:
            w: a word

        Returns:
            embedding vector
        """

    def _encode(self, tokens: List[str], mean: bool) -> Union[List[np.ndarray], np.ndarray]:
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
                    emb = self._get_word_vector(t)
                except KeyError:
                    emb = np.zeros(self.dim, dtype=np.float32)
                self.tok2emb[t] = emb
            embedded_tokens.append(emb)

        if mean is None:
            mean = self.mean

        if mean:
            filtered = [et for et in embedded_tokens if np.any(et)]
            if filtered:
                return np.mean(filtered, axis=0)
            return np.zeros(self.dim, dtype=np.float32)

        return embedded_tokens
