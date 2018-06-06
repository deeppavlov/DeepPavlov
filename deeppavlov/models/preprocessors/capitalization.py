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
import re

from deeppavlov.core.models.component import Component
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.common.registry import register
# from deeppavlov.models.tokenizers import nltk_tokenizer


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


def process_word(word, to_lower=False, append_case=None):
    if all(x.isupper() for x in word) and len(word) > 1:
        uppercase = "<ALL_UPPER>"
    elif word[0].isupper():
        uppercase = "<FIRST_UPPER>"
    else:
        uppercase = None
    if to_lower:
        word = word.lower()
    if word.isdigit():
        answer = ["<DIGIT>"]
    elif word.startswith("http://") or word.startswith("www."):
        answer = ["<HTTP>"]
    else:
        answer = list(word)
    if to_lower and uppercase is not None:
        if append_case == "first":
            answer = [uppercase] + answer
        elif append_case == "last":
            answer = answer + [uppercase]
    return tuple(answer)


@register('lowercase_preprocessor')
class LowercasePreprocessor(Component):

    def __init__(self, to_lower=True, append_case="first", *args, **kwargs):
        self.to_lower = to_lower
        self.append_case = append_case

    def __call__(self, tokens_batch, **kwargs):
        answer = []
        for elem in tokens_batch:
            if isinstance(elem, str):
                elem = [x for x in re.split("(\w+|[,.])", elem) if x.strip() != ""]
            answer.append([process_word(x, self.to_lower, self.append_case) for x in elem])
        return answer
