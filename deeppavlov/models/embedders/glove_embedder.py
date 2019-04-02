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
import pickle
from logging import getLogger
from typing import Iterator

import numpy as np
from gensim.models import KeyedVectors
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.models.embedders.abstract_embedder import Embedder

log = getLogger(__name__)


@register('glove')
class GloVeEmbedder(Embedder):
    """
    Class implements GloVe embedding model

    Args:
        load_path: path where to load pre-trained embedding model from
        pad_zero: whether to pad samples or not

    Attributes:
        model: GloVe model instance
        tok2emb: dictionary with already embedded tokens
        dim: dimension of embeddings
        pad_zero: whether to pad sequence of tokens with zeros or not
        load_path: path with pre-trained GloVe model
    """

    def _get_word_vector(self, w: str) -> np.ndarray:
        return self.model[w]

    def load(self) -> None:
        """
        Load dict of embeddings from given file
        """
        log.info(f"[loading GloVe embeddings from `{self.load_path}`]")
        if not self.load_path.exists():
            log.warning(f'{self.load_path} does not exist, cannot load embeddings from it!')
            return
        self.model = KeyedVectors.load_word2vec_format(str(self.load_path))
        self.dim = self.model.vector_size

    @overrides
    def __iter__(self) -> Iterator[str]:
        """
        Iterate over all words from GloVe model vocabulary

        Returns:
            iterator
        """
        yield from self.model.vocab

    def serialize(self) -> bytes:
        return pickle.dumps(self.model, protocol=4)

    def deserialize(self, data: bytes) -> None:
        self.model = pickle.loads(data)
        self.dim = self.model.vector_size
