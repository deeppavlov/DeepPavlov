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
from typing import List, Union

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)

def preprocess_scores(scores, is_binary, class_id: int = 1):
    scores = np.array(scores)
    return scores if is_binary else scores[:, class_id]

@register('dnnc_proba2labels')
class Proba2Labels(Component):

    def __init__(self,
                 confidence_threshold: float = 0.0,
                 pooling: str = 'max',
                 is_binary: bool = False,
                 **kwargs) -> None:

        self.confidence_threshold = confidence_threshold
        self.pooling = pooling
        self.is_binary = is_binary


    def __call__(self,
                 simmilarity_scores: Union[np.ndarray, List[List[float]], List[List[int]]],
                 x: List[str],
                 x_populated: List[str],
                 x_support: List[str],
                 y_support: List[str]):
        y_pred = []

        simmilarity_scores = preprocess_scores(simmilarity_scores, self.is_binary)
        x_populated = np.array(x_populated)
        x_support = np.array(x_support)
        y_support = np.array(y_support)

        unique_labels = np.unique(y_support)

        for example in x: 
            example_mask = np.where(np.logical_xor(x_populated == example, x_support == example))
            example_simmilarity_scores = simmilarity_scores[example_mask]
            example_y_support = y_support[example_mask]

            probability_by_label = []
            for label in unique_labels:
                label_mask = np.where(example_y_support == label)
                label_simmilarity_scores = example_simmilarity_scores[label_mask]
                if self.pooling == 'avg':
                    label_probability = np.mean(label_simmilarity_scores)
                elif self.pooling == 'max':
                    label_probability = np.max(label_simmilarity_scores)
                probability_by_label.append(label_probability)
            
            probability_by_label = np.array(probability_by_label)
            max_probability = max(probability_by_label)
            max_probability_label = unique_labels[np.argmax(probability_by_label)]
            prediction = "oos" if max_probability < self.confidence_threshold else max_probability_label
            
            y_pred.append(prediction)
    
        return y_pred
