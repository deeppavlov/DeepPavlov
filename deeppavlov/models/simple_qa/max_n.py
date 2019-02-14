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
from typing import List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from pathlib import Path


@register('max_n')
class Max_n(Component):
    '''
        the class gets n classes with maximum probabilities using probabilities of
        classes from the relation prediction model
    '''

    def __init__(self, top_k_classes: int, classes_vocab_keys: Tuple, *args, **kwargs) -> None:
        self.top_k_classes = top_k_classes
        self.classes = list(classes_vocab_keys)

    def __call__(self, probas_batch: List[List[float]], *args, **kwargs) -> List[List[str]]:
        max_n_batch = []
        for probas in probas_batch:
            max_n = np.asarray(probas).argsort()[-self.top_k_classes:][::-1]  # Make it top n and n to the __init__
            max_n_classes = [self.classes[num] for num in max_n]
            max_n_batch.append(max_n_classes)
        for c, p in zip(max_n, max_n_classes):
            print(c, p)
        return max_n_batch
