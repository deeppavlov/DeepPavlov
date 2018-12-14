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

from typing import List

import pymorphy2

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register("pymorphy_pos_tagger")
class PyPOSTagger(Component):
    def __init__(self, **kwargs):
        self.pos = pymorphy2.MorphAnalyzer()

    def __call__(self, tokens_batch: List[List[str]], *args, **kwargs) -> List[List[str]]:
        """Takes batch of tokens and returns the batch of POS tags."""
        pos_batch = []
        for utterance in tokens_batch:
            pos_utterance = []
            for token in utterance:
                p = self.pos.parse(token)[0]
                pos_utterance.append(p.tag.POS)
            pos_batch.append(pos_utterance)
        return pos_batch
