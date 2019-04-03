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
import string
from logging import getLogger
from math import log10
from typing import Iterable, List, Tuple, Optional

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from .levenshtein_searcher import LevenshteinSearcher

logger = getLogger(__name__)


@register('spelling_levenshtein')
class LevenshteinSearcherComponent(Component):
    """Component that finds replacement candidates for tokens at a set Damerau-Levenshtein distance

    Args:
        words: list of every correct word
        max_distance: maximum allowed Damerau-Levenshtein distance between source words and candidates
        error_probability: assigned probability for every edit
        vocab_penalty: assigned probability of an out of vocabulary token being the correct one without changes

    Attributes:
        max_distance: maximum allowed Damerau-Levenshtein distance between source words and candidates
        error_probability: assigned logarithmic probability for every edit
        vocab_penalty: assigned logarithmic probability of an out of vocabulary token being the correct one without
         changes
    """

    _punctuation = frozenset(string.punctuation)

    def __init__(self, words: Iterable[str], max_distance: int = 1, error_probability: float = 1e-4,
                 vocab_penalty: Optional[float] = None, **kwargs):
        words = list({word.strip().lower().replace('ё', 'е') for word in words})
        alphabet = sorted({letter for word in words for letter in word})
        self.max_distance = max_distance
        self.error_probability = log10(error_probability)
        self.vocab_penalty = self.error_probability if vocab_penalty is None else log10(vocab_penalty)
        self.searcher = LevenshteinSearcher(alphabet, words, allow_spaces=True, euristics=2)

    def _infer_instance(self, tokens: Iterable[str]) -> List[List[Tuple[float, str]]]:
        candidates = []
        for word in tokens:
            if word in self._punctuation:
                candidates.append([(0, word)])
            else:
                c = {candidate: self.error_probability * distance
                     for candidate, distance in self.searcher.search(word, d=self.max_distance)}
                c[word] = c.get(word, self.vocab_penalty)
                candidates.append([(score, candidate) for candidate, score in c.items()])
        return candidates

    def __call__(self, batch: Iterable[Iterable[str]], *args, **kwargs) -> List[List[List[Tuple[float, str]]]]:
        """Propose candidates for tokens in sentences

        Args:
            batch: batch of tokenized sentences

        Returns:
            batch of lists of probabilities and candidates for every token
        """
        return [self._infer_instance(tokens) for tokens in batch]
