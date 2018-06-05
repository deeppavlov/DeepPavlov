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
from math import log10
from typing import Iterable

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger

from .levenstein_searcher import LevensteinSearcher


logger = get_logger(__name__)


@register('spelling_levenstein')
class LevensteinSearcherComponent(Component):
    def __init__(self, words: Iterable[str], max_distance=1, error_probability=1e-4, *args, **kwargs):
        """

        :param words: list of every correct word
        :param max_distance: maximum allowed Damerau-Levenstein distance between source words and candidates
        :param error_probability: assigned probability for every edit
        """
        words = list({word.strip().lower().replace('ё', 'е') for word in words})
        alphabet = sorted({letter for word in words for letter in word})
        self.max_distance = max_distance
        self.error_probability = log10(error_probability)
        self.vocab_penalty = self.error_probability * 2
        self.searcher = LevensteinSearcher(alphabet, words, allow_spaces=True, euristics=2)

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
