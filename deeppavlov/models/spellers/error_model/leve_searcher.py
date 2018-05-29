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

import csv
from typing import List
from collections import defaultdict, Counter
from heapq import heappop, heappushpop, heappush
from math import log, exp

from tqdm import tqdm

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger

from .searcher.levenstein_searcher import LevensteinSearcher


logger = get_logger(__name__)


@register('spelling_levenstein_searcher')
class LeveSearcher(Component):
    def __init__(self, words, allow_spaces=True, max_distance=1, error_probability=1e-4, *args, **kwargs):
        words = list(words)
        alphabet = sorted({letter for word in words for letter in word})
        self.max_distance = max_distance
        self.error_probability = error_probability
        self.vocab_penalty = self.error_probability * 2
        self.searcher = LevensteinSearcher(alphabet, words, allow_spaces=allow_spaces, euristics=2)

    def _infer_instance(self, tokens):
        candidates = []
        for word in tokens:
            c = {candidate: self.error_probability * distance
                 for candidate, distance in self.searcher.search(word, d=self.max_distance)}
            c[word] = c.get(word, self.vocab_penalty)
            candidates.append([(score, candidate) for candidate, score in c.items()])
        return candidates

    def __call__(self, batch, *args, **kwargs):
        return [self._infer_instance(tokens) for tokens in batch]
