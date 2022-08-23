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
import torch

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('proba2labels_few_shot')
class Proba2LabelsFewShot(Component):
    """
    Class implements probability to labels processing using the following ways: \
     choosing one or top_n indices with maximal probability or choosing any number of indices \
      which probabilities to belong with are higher than given confident threshold

    Args:
        max_proba: whether to choose label with maximal probability
        confidence_threshold: boundary probability value for sample to belong with the class (best use for multi-label)
        top_n: how many top labels with the highest probabilities to return

    Attributes:
        max_proba: whether to choose label with maximal probability
        confidence_threshold: boundary probability value for sample to belong with the class (best use for multi-label)
        top_n: how many top labels with the highest probabilities to return
    """

    def __init__(self,
                 max_proba: bool = None,
                 confidence_threshold: float = None,
                 top_n: int = None,
                 is_binary: bool = False,
                 pooling: str = 'mean',
                 **kwargs) -> None:
        """ Initialize class with given parameters"""

        self.confidence_threshold = confidence_threshold
        self.is_binary = is_binary
        self.pooling = pooling

    def __call__(self,
                 data: Union[np.ndarray,
                             List[List[float]],
                             List[List[int]]],
                 train_cat: List[str],
                 test_cat: List[str],
                 *args,
                 **kwargs):
        """
        Process probabilities to labels

        Args:
            data: list of vectors with probability distribution

        Returns:
            list of labels
        """
        probas_by_class = []

        unique_cats = list(sorted(set(train_cat)))
        train_cat = np.array(train_cat)
        
        for cat in unique_cats:
            ind_mask = np.where(train_cat == cat)

            if self.pooling == 'mean':
                class_proba = np.mean(data[ind_mask])
            elif self.pooling == 'max':
                class_proba = np.max(data[ind_mask])

            probas_by_class.append(class_proba)

        if self.confidence_threshold:
            max_conf = np.max(probas_by_class)

            if max_conf > self.confidence_threshold:
                pred_id = np.argmax(probas_by_class)
                y_pred = unique_cats[pred_id]
            else:
                y_pred = 'oos'

        return [[y_pred], [test_cat[0]]]
        

        