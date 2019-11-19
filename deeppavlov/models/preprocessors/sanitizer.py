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
import sys
import unicodedata

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('sanitizer')
class Sanitizer(Component):
    """Remove all combining characters like diacritical marks from tokens

    Args:
        diacritical: whether to remove diacritical signs or not
            diacritical signs are something like hats and stress marks
        nums: whether to replace all digits with 1 or not
    """

    def __init__(self,
                 diacritical: bool = True,
                 nums: bool = False,
                 *args, **kwargs) -> None:
        self.diacritical = diacritical
        self.nums = nums
        self.combining_characters = dict.fromkeys([c for c in range(sys.maxunicode)
                                                   if unicodedata.combining(chr(c))])

    def filter_diacritical(self, tokens_batch):
        """Takes batch of tokens and returns the batch with sanitized tokens"""
        sanitized_batch = []
        for utterance in tokens_batch:
            sanitized_utterance = []
            for token in utterance:
                token = unicodedata.normalize('NFD', token)
                sanitized_utterance.append(token.translate(self.combining_characters))
            sanitized_batch.append(sanitized_utterance)
        return sanitized_batch

    def replace_nums(self, tokens_batch):
        sanitized_batch = []
        for utterance in tokens_batch:
            sanitized_batch.append([re.sub('[0-9]', '1', token) for token in utterance])
        return sanitized_batch

    def __call__(self, tokens_batch, **kwargs):
        if self.filter_diacritical:
            tokens_batch = self.filter_diacritical(tokens_batch)
        if self.nums:
            tokens_batch = self.replace_nums(tokens_batch)
        return tokens_batch
