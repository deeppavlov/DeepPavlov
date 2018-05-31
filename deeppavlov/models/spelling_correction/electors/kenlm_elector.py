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

from typing import List, Tuple

import kenlm

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger


logger = get_logger(__name__)


@register('kenlm_elector')
class KenlmElector(Component):
    def __init__(self, load_path, beam_size=4, *args, **kwargs):
        self.lm = kenlm.Model(str(expand_path(load_path)))
        self.beam_size = beam_size

    def __call__(self, batch: List[List[List[Tuple[float, str]]]], *args, **kwargs):
        return [self._infer_instance(candidates) for candidates in batch]

    def _infer_instance(self, candidates: List[List[Tuple[float, str]]]):
        candidates = candidates + [[(0, '</s>')]]
        state = kenlm.State()
        self.lm.BeginSentenceWrite(state)
        beam = [(0, state, [])]
        for sublist in candidates:
            new_beam = []
            for beam_score, beam_state, beam_words in beam:
                for score, candidate in sublist:
                    prev_state = beam_state
                    c_score = 0
                    cs = candidate.split()
                    for candidate in cs:
                        state = kenlm.State()
                        c_score += self.lm.BaseScore(prev_state, candidate, state)
                        prev_state = state
                    new_beam.append((beam_score + score + c_score, state, beam_words + cs))
            new_beam.sort(reverse=True)
            beam = new_beam[:self.beam_size]
        score, state, words = beam[0]
        return words[:-1]
