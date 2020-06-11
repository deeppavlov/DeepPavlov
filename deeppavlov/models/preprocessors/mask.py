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

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('mask')
class Mask(Component):
    """Takes a batch of tokens and returns the masks of corresponding length"""
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def __call__(tokens_batch, **kwargs):
        batch_size = len(tokens_batch)
        max_len = max(len(utt) for utt in tokens_batch)
        mask = np.zeros([batch_size, max_len], dtype=np.float32)
        for n, utterance in enumerate(tokens_batch):
            mask[n, :len(utterance)] = 1

        return mask
