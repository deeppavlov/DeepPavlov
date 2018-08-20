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

from overrides import overrides
from typing import List, Union
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


@register('avr_emb')
class AvrEmb(Component):
    """
        Parameters:

        Examples:

        """

    def __init__(self, **kwargs):
        self.tokenizer = kwargs.get('tokenizer', None)
        self.vec_ = kwargs.get('vectorizer', None)
        self.vec = self.vec_.vectorizer

    @overrides
    def __call__(self, data: List[Union[list, np.ndarray]], text: List[List[str]], *args, **kwargs) -> List:
        result = []
        if self.vec is None:
            vec = self.average(data, result)
        else:
            vec = self.weigh_tfidf(data, text, result)
        return vec

    def average(self, data, res):
        for x in data:
            res.append(np.average(np.array(x), axis=0))
        return self.result

    def weigh_tfidf(self, data, text, res):
        feature_names = self.vec.get_feature_names()
        x_test = self.vec.transform(text)
        text = self.tokenizer(text)

        for i in range(len(text)):
            info = {}
            feature_index = x_test[i, :].nonzero()[1]
            tfidf_scores = zip(feature_index, [x_test[i, x] for x in feature_index])

            for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
                info[w] = s

            weights = np.array([info[w] if w in info.keys() else 0.05 for w in text[i]])
            matrix = np.array(data[i])

            res.append(np.dot(weights, matrix))
        return res
