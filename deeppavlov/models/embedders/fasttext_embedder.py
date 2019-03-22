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

from logging import getLogger
from typing import Iterator

import fastText
import numpy as np
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.models.embedders.abstract_embedder import Embedder

log = getLogger(__name__)


@register('fasttext')
class FasttextEmbedder(Embedder):
    """
    Class implements fastText embedding model

    Args:
        load_path: path where to load pre-trained embedding model from
        pad_zero: whether to pad samples or not

    Attributes:
        model: fastText model instance
        tok2emb: dictionary with already embedded tokens
        dim: dimension of embeddings
        pad_zero: whether to pad sequence of tokens with zeros or not
        load_path: path with pre-trained fastText binary model
    """

    def _get_word_vector(self, w: str) -> np.ndarray:
        return self.model.get_word_vector(w)

    def load(self) -> None:
        """
        Load fastText binary model from self.load_path
        """
        log.info(f"[loading fastText embeddings from `{self.load_path}`]")
        self.model = fastText.load_model(str(self.load_path))
        self.dim = self.model.get_dimension()

    @overrides
    def __iter__(self) -> Iterator[str]:
        """
        Iterate over all words from fastText model vocabulary

        Returns:
            iterator
        """
        yield from self.model.get_words()
