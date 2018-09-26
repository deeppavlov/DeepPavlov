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

import re
from typing import List, Union

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('data_sum_operator')
class DataSumOperator(Component):
    """
    Class implements summation of given elements for every smaple in a batch
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> List[str]:
        """
        Preprocess given batch

        Args:
            batch: list of text samples
            **kwargs: additional arguments

        Returns:
            list of preprocessed text samples
        """
        res = []

        for i in range(len(args[0])):
            sample = args[0][i]
            for j in range(1, len(args)):
                sample += args[j][i]
            res.append(sample)

        return res
