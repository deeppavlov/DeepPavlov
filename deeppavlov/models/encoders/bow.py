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

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('bow')
class BoW_encoder(Component):
    def __init__(self, save_path=None, **kwargs):
        super().__init__(save_path=save_path)

    def _encode(self, utterance, vocab):
        bow = np.zeros([len(vocab)], dtype=np.int32)
        for word in utterance.split(' '):
            if word in vocab:
                idx = vocab[word]
                bow[idx] += 1
        return bow

    def __call__(self, utterance, vocab, *args):
        return self._encode(utterance, vocab)
