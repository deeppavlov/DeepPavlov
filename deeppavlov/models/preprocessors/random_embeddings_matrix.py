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

import numpy as np

from deeppavlov.core.common.registry import register


@register('random_emb_mat')
class RandomEmbeddingsMatrix:
    """Assembles matrix of random embeddings.

    Args:
        vocab_len: length of the vocabulary (number of tokens in it)
        emb_dim: dimensionality of the embeddings

    Attributes:
        dim: dimensionality of the embeddings
    """

    def __init__(self, vocab_len: int, emb_dim: int, *args, **kwargs) -> None:
        self.emb_mat = np.random.randn(vocab_len, emb_dim).astype(np.float32) / np.sqrt(emb_dim)

    @property
    def dim(self):
        return self.emb_mat.shape[1]
