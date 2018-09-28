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

from typing import List, Tuple, Union
import numpy as np

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component

log = get_logger(__name__)


@register('proba2labels')
class Proba2Labels(Component):
    """
    Class implements probability to labels processing using two different ways: \
     choosing label with maximal probability or choosing any number of labels \
      which probabilities to belong with are higher than given confident threshold

    Args:
        max_proba: whether to choose label with maximal probability
        confident_threshold: boundary probability value for smaple to belong with the class (best use for multi-label)

    Attributes:
        max_proba: whether to choose label with maximal probability
        confident_threshold: boundary probability value for smaple to belong with the class (best use for multi-label)
    """

    def __init__(self,
                 max_proba: bool = None,
                 confident_threshold: float = None,
                 **kwargs) -> None:
        """ Initialize class with given parameters"""

        self.max_proba = max_proba
        self.confident_threshold = confident_threshold

    def __call__(self, data: Union[List[List[float]], np.ndarray],
                 *args, **kwargs) -> Union[List[List[int]], List[int]]:
        """
        Process probabilities to labels

        Args:
            data: list of vectors with probability distribution
            *args:
            **kwargs:

        Returns:
            list of labels (only label classification) or list of lists of labels (multi-label classification)
        """
        if self.confident_threshold:
            return [list(np.array(data[i])[np.where(np.array(data[i]) > self.confident_threshold)[0]])
                    for i in range(len(data))]
        elif self.max_proba:
            return list(np.argmax(data, axis=1))
        else:
            ConfigError("Proba2Labels requires one of two arguments: bool `max_proba` or "
                        "float `confident_threshold` for multi-label classification")
