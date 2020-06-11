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

from typing import List

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('bow')
class BoWEmbedder(Component):
    """
    Performs one-hot encoding of tokens based on a pre-built vocabulary of tokens.

    Parameters:
        depth: size of output numpy vector.
        with_counts: flag denotes whether to use binary encoding (with zeros and ones),
            or to use counts as token representation.

    Example:
        .. code:: python

            >>> bow = BoWEmbedder(depth=3)

            >>> bow([[0, 1], [1], [])
            [array([1, 1, 0], dtype=int32),
             array([0, 1, 0], dtype=int32),
             array([0, 0, 0], dtype=int32)]
    """

    def __init__(self, depth: int, with_counts: bool = False, **kwargs) -> None:
        self.depth = depth
        self.with_counts = with_counts

    def _encode(self, token_indices: List[int]) -> np.ndarray:
        bow = np.zeros([self.depth], dtype=np.int32)
        for idx in token_indices:
            if self.with_counts:
                bow[idx] += 1
            else:
                bow[idx] = 1
        return bow

    def __call__(self, batch: List[List[int]]) -> List[np.ndarray]:
        return [self._encode(sample) for sample in batch]
