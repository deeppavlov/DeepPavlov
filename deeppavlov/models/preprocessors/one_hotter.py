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

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad

import numpy as np


@register('one_hotter')
class OneHotter(Component):
    """
    One-hot featurizer with zero-padding.

    Parameters:
        depth: the depth for one-hotting
        pad_zeros: whether to pad elements of batch with zeros
    """
    def __init__(self, depth: int, pad_zeros: bool = True, *args, **kwargs):
        self._depth = depth
        self._pad_zeros = pad_zeros

    def __call__(self, batch, **kwargs):
        one_hotted_batch = []
        for utt in batch:
            one_hotted_utt = self._to_one_hot(utt, self._depth)
            one_hotted_batch.append(one_hotted_utt)
        if self._pad_zeros:
            one_hotted_batch = zero_pad(one_hotted_batch)
        return one_hotted_batch

    @staticmethod
    def _to_one_hot(x, n):
        b = np.zeros([len(x), n], dtype=np.float32)
        for q, tok in enumerate(x):
            b[q, tok] = 1
        return b
