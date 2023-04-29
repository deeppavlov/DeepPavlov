# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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
from ufal.chu_liu_edmonds import chu_liu_edmonds

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('chu_liu_edmonds_transformer')
class ChuLiuEdmonds(Component):
    """
    A wrapper for Chu-Liu-Edmonds algorithm for maximum spanning tree
    """

    def __init__(self, min_edge_prob=1e-6, **kwargs):
        self.min_edge_prob = min_edge_prob

    def __call__(self, probs: List[np.ndarray]) -> List[List[int]]:
        """Applies Chu-Liu-Edmonds algorithm to the matrix of head probabilities.
        probs: a 3D-array of probabilities of shape B*L*(L+1)
        """
        answer = []
        for elem in probs:
            m, n = elem.shape
            if n == m + 1:
                elem = np.log10(np.maximum(self.min_edge_prob, elem)) - np.log10(self.min_edge_prob)
                elem = np.concatenate([np.zeros_like(elem[:1, :]), elem], axis=0)
                # it makes impossible to create multiple edges 0->i
                elem[1:, 0] += np.log10(self.min_edge_prob) * len(elem)
                heads, _ = chu_liu_edmonds(elem.astype("float64"))
                answer.append(heads[1:])
            else:
                raise ValueError("First and second axis lengths m, n of probs should satisfy the condition n == m + 1")
        return answer
