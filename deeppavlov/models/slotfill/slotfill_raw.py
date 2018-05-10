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
import json
from math import exp

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import tokenize_reg
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from overrides import overrides

log = get_logger(__name__)


@register('slotfill_raw')
class SlotFillingComponent(Component, Serializable):
    def __init__(self, threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        # self._slot_vals is the dictionary of slot values
        self._slot_vals = None
        self.load()

    @overrides
    def __call__(self, batch, *args, **kwargs):
        if isinstance(batch[0], str):
            batch = [tokenize_reg(instance.strip()) for instance in batch]

        slots = [{}] * len(batch)

        m = [i for i, v in enumerate(batch) if v]
        if m:
            batch = [batch[i] for i in m]
            # tags_batch = self._ner_network.predict_for_token_batch(batch)
            # batch example: [['is', 'there', 'anything', 'else']]
            for i, tokens in zip(m, batch):
                # tokens are['is', 'there', 'anything', 'else']
                slots[i] = self._predict_slots(tokens, self.threshold)
                # slots[i] example {'food': 'steakhouse'}
        # slots we want, example : [{'pricerange': 'moderate', 'area': 'south'}]
        return slots

    def _predict_slots(self, tokens, threshold):
        # For utterance extract named entities and perform normalization for slot filling
        entities, slots = self._fuzzy_finder(self._slot_vals, tokens, threshold)
        slot_values = {}
        for entity, slot in zip(entities, slots):
            slot_values[slot] = entity
        return slot_values

    def load(self, *args, **kwargs):
        with open(self.load_path) as f:
            self._slot_vals = json.load(f)

    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self._slot_vals, f)

    def _fuzzy_finder(self, slot_dict, tokens, threshold):
        global input_entity
        if isinstance(tokens, list):
            input_entity = ' '.join(tokens)
        entities = []
        slots = []
        for slot, tag_dict in slot_dict.items():
            r, candidate_entity = self.get_candidate(input_entity, tag_dict, self.get_ratio)
            if r > threshold:
                entities.append(candidate_entity)
                slots.append(slot)
        return entities, slots

    def get_candidate(self, input_text, tag_dict, score_function):
        r = -1
        candidate = ""
        for entity_name, entity_list in tag_dict.items():
            for entity in entity_list:
                ratio = score_function(entity.lower(), input_text.lower())
                if ratio < r:
                    continue
                if ratio > r:
                    r = ratio
                    candidate = entity_name
        return r, candidate

    def get_ratio(self, needle, haystack):
        d = self.fuzzy_substring_distance(needle, haystack)
        m = len(needle) - d
        return exp(-d / 5) * (m / len(needle))

    @staticmethod
    def fuzzy_substring_distance(needle, haystack):
        """Calculates the fuzzy match of needle in haystack,
        using a modified version of the Levenshtein distance
        algorithm.
        The function is modified from the Levenshtein function
        in the bktree module by Adam Hupp
        :type needle: string
        :type haystack: string"""
        m, n = len(needle), len(haystack)

        # base cases
        if m == 1:
            return needle not in haystack
        if not n:
            return m

        row1 = [0] * (n + 1)
        for j in range(0, n + 1):
            if j == 0 or not haystack[j - 1].isalnum():
                row1[j] = 0
            else:
                row1[j] = row1[j - 1] + 1

        for i in range(0, m):
            row2 = [i + 1]
            for j in range(0, n):
                cost = (needle[i] != haystack[j])
                row2.append(min(row1[j + 1] + 1, row2[j] + 1, row1[j] + cost))
            row1 = row2

        d = n + m
        for j in range(0, n + 1):
            if j == 0 or j == n or not haystack[j].isalnum():
                d = min(d, row1[j])
        return d
