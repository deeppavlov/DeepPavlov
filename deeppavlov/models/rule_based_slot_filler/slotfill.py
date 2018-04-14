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
from pathlib import Path

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import tokenize_reg
from deeppavlov.core.models.component import Component
from overrides import overrides

log = get_logger(__name__)


@register('simple_slotfilling')
class SlotFillingComponent(Component):
    def __init__(self, **kwargs):
        super().__init__()
        self.threshold = kwargs.pop('threshold', 0.9)
        save_path = kwargs.pop('save_path', "slots")
        file_name = kwargs.pop('file_name', "slot_vals.json")
        self.save_path = expand_path(save_path)
        slot_vals_filepath = Path(self.save_path) / file_name

        # self._slot_vals is the dictionary of slot values
        try:
            with open(slot_vals_filepath) as f:
                self._slot_vals = json.load(f)
        except IOError as ioe:
            print(ioe)
            self._slot_vals = {}

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
        entities, slots = _fuzzy_finder(self._slot_vals, tokens, threshold)
        slot_values = {}
        for entity, slot in zip(entities, slots):
            slot_values[slot] = entity
        return slot_values


def _fuzzy_finder(slot_dict, tokens, threshold):
    global input_entity
    if isinstance(tokens, list):
        input_entity = ' '.join(tokens)
    entities = []
    slots = []
    for slot, tag_dict in slot_dict.items():
        r, candidate_entity = get_candidate(input_entity, tag_dict, get_ratio)
        if r > threshold:
            entities.append(candidate_entity)
            slots.append(slot)
    return entities, slots


def get_candidate(input_text, tag_dict, score_function):
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


def get_ratio(needle, haystack):
    d = fuzzy_substring_distance(needle, haystack)
    m = len(needle) - d
    return exp(-d / 5) * (m / len(needle))


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
