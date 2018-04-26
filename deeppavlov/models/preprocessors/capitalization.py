"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from deeppavlov.core.models.component import Component
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.common.registry import register
import numpy as np


@register('capitalization_featurizer')
class CapitalizationPreprocessor(Component):
    """ Patterns:
        - no capitals
        - single capital single character
        - single capital multiple characters
        - all capitals multiple characters
    """
    def __init__(self, pad_zeros=True, *args, **kwargs):
        self.pad_zeros = pad_zeros
        self._num_of_features = 4

    @property
    def dim(self):
        return self._num_of_features

    def __call__(self, tokens_batch, **kwargs):
        cap_batch = []
        max_batch_len = 0
        for utterance in tokens_batch:
            cap_list = []
            max_batch_len = max(max_batch_len, len(utterance))
            for token in utterance:
                cap = np.zeros(4, np.float32)
                # Check the case and produce corresponding one-hot
                if len(token) > 0:
                    if token[0].islower():
                        cap[0] = 1
                    elif len(token) == 1 and token[0].isupper():
                        cap[1] = 1
                    elif len(token) > 1 and token[0].isupper() and any(ch.islower() for ch in token):
                        cap[2] = 1
                    elif all(ch.isupper() for ch in token):
                        cap[3] = 1
                cap_list.append(cap)
            cap_batch.append(cap_list)
        if self.pad_zeros:
            return zero_pad(cap_batch)
        else:
            return cap_batch
