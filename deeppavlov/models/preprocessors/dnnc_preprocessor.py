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
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('dnnc_pair_generator')
class PairGenerator(Component):
    """
    Generates all possible ordered pairs from 'texts_batch' and 'support_dataset'
    
    Args:
        bidirectional: adds pairs in reverse order
    """

    def __init__(self, bidirectional: bool = False) -> None:
        self.bidirectional = bidirectional

    def __call__(self,
                 texts_batch: List[str],
                 support_dataset: List[List[str]]
                 ) -> Tuple[List[str], List[str], List[str], List[str]]:
        hypotesis_batch = []
        premise_batch = []
        hypotesis_labels_batch = []
        for [premise, [hypotesis, hypotesis_labels]] in zip(texts_batch * len(support_dataset),
                                                            np.repeat(support_dataset, len(texts_batch), axis=0)):
            premise_batch.append(premise)
            hypotesis_batch.append(hypotesis)
            hypotesis_labels_batch.append(hypotesis_labels)

            if self.bidirectional:
                premise_batch.append(hypotesis)
                hypotesis_batch.append(premise)
                hypotesis_labels_batch.append(hypotesis_labels)
        return texts_batch, hypotesis_batch, premise_batch, hypotesis_labels_batch
