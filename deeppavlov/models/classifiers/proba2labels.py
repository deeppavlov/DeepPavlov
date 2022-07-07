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

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('proba2labels')
class Proba2Labels(Component):
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
                 **kwargs) -> None:
        """ Initialize class with given parameters"""

        self.max_proba = max_proba
        self.confidence_threshold = confidence_threshold
        self.top_n = top_n
        self.is_binary = is_binary

    def __call__(self,
                 data: Union[np.ndarray,
                             List[List[float]],
                             List[List[int]]],
                 *args,
                 **kwargs) -> Union[List[List[int]], List[int]]:
        """
        Process probabilities to labels

        Args:
            data: list of vectors with probability distribution

        Returns:
            list of labels (only label classification) or list of lists of labels (multi-label classification)
        """
        if self.confidence_threshold:
            if self.is_binary:
                return [int(el > self.confidence_threshold) for el in data]
            else:
                return [list(np.where(np.array(d) > self.confidence_threshold)[0])
                        for d in data]
        elif self.max_proba:
            return [np.argmax(d) for d in data]
        elif self.top_n:
            return [np.argsort(d)[::-1][:self.top_n] for d in data]
        else:
            raise ConfigError("Proba2Labels requires one of three arguments: bool `max_proba` or "
                              "float `confidence_threshold` for multi-label classification or"
                              "integer `top_n` for choosing several labels with the highest probabilities")
