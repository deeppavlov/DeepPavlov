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
from typing import List

from core.models.serializable import Serializable
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from pathlib import Path


@register('max_n')
class Max_n(Component, Serializable):
    '''
        the class gets n classes with maximum probabilities using probabilities of
        classes from the relation prediction model
    '''

    def __init__(self, n_maximum_classes: int, load_path: str, *args, **kwargs) -> None:
        super().__init__(save_path = None, load_path = load_path)
        self.n_maximum_classes = n_maximum_classes
        self.load()

    def load(self) -> None:
        load_path = Path(self.load_path).expanduser()
        with open(load_path, 'r') as fl:
            lines = fl.readlines()
            self.classes = [line.split('\t')[0] for line in lines]

    def save(self):
        pass

    def __call__(self, probas_batch: List[List[float]], *args, **kwargs) -> List[List[str]]:
        max_n_batch = []

        for probas in probas_batch:
            max_n = np.asarray(probas).argsort()[-5:][::-1]  # Make it top n and n to the __init__
            max_n_classes = [self.classes[num] for num in max_n]
            max_n_batch.append(max_n_classes)
        
        return max_n_batch
