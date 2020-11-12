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

import json
import tempfile
from collections import defaultdict
from logging import getLogger
from math import exp

from pathlib import Path
from overrides import overrides

from deeppavlov.core.common.file import read_yaml
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.dataset_readers.md_yaml_dialogs_reader import MD_YAML_DialogsDatasetReader, DomainKnowledge

log = getLogger(__name__)


@register('slotfill_raw')
class SlotFillingComponent(Component, Serializable):
    """Slot filling using Fuzzy search"""

    def __init__(self, threshold: float = 0.7, return_all: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.return_all = return_all
        # self._slot_vals is the dictionary of slot values
        self._slot_vals = None
        self.load()

    @overrides
    def __call__(self, batch, *args, **kwargs):
        slots = [{}] * len(batch)

        m = [i for i, v in enumerate(batch) if v]
        if m:
            batch = [batch[i] for i in m]
            # tags_batch = self._ner_network.predict_for_token_batch(batch)
            # batch example: [['is', 'there', 'anything', 'else']]
            for i, tokens in zip(m, batch):
                # tokens are['is', 'there', 'anything', 'else']
                slots_values_lists = self._predict_slots(tokens)
                if self.return_all:
                    slots[i] = dict(slots_values_lists)
                else:
                    slots[i] = {slot: val_list[0] for slot, val_list in slots_values_lists.items()}
                # slots[i] example {'food': 'steakhouse'}
        # slots we want, example : [{'pricerange': 'moderate', 'area': 'south'}]
        return slots

    def _predict_slots(self, tokens):
        # For utterance extract named entities and perform normalization for slot filling
        entities, slots = self._fuzzy_finder(self._slot_vals, tokens)
        slot_values = defaultdict(list)
        for entity, slot in zip(entities, slots):
            slot_values[slot].append(entity)
        return slot_values

    def load(self, *args, **kwargs):
        with open(self.load_path, encoding='utf8') as f:
            self._slot_vals = json.load(f)

    def deserialize(self, data):
        self._slot_vals = json.loads(data)

    def save(self):
        with open(self.save_path, 'w', encoding='utf8') as f:
            json.dump(self._slot_vals, f)

    def serialize(self):
        return json.dumps(self._slot_vals)

    def _fuzzy_finder(self, slot_dict, tokens):
        global input_entity
        if isinstance(tokens, list):
            input_entity = ' '.join(tokens)
        entities = []
        slots = []
        for slot, tag_dict in slot_dict.items():
            candidates = self.get_candidate(input_entity, tag_dict, self.get_ratio)
            for candidate in candidates:
                if candidate not in entities:
                    entities.append(candidate)
                    slots.append(slot)
        return entities, slots

    def get_candidate(self, input_text, tag_dict, score_function):
        candidates = []
        positions = []
        for entity_name, entity_list in tag_dict.items():
            for entity in entity_list:
                ratio, j = score_function(entity.lower(), input_text.lower())
                if ratio >= self.threshold:
                    candidates.append(entity_name)
                    positions.append(j)
        if candidates:
            _, candidates = list(zip(*sorted(zip(positions, candidates))))
        return candidates

    def get_ratio(self, needle, haystack):
        d, j = self.fuzzy_substring_distance(needle, haystack)
        m = len(needle) - d
        return exp(-d / 5) * (m / len(needle)), j

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
            not_found = needle not in haystack
            not_found = float(not_found)  # float required by the method usage
            occurrence_ix = 0 if not_found else haystack.index(needle)
            return not_found, occurrence_ix
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
        j_min = 0
        for j in range(0, n + 1):
            if j == 0 or j == n or not haystack[j].isalnum():
                if d > row1[j]:
                    d = row1[j]
                    j_min = j
                # d = min(d, row1[j])
        return d, j_min


@register('slotfill_raw_rasa')
class RASA_SlotFillingComponent(SlotFillingComponent):
    """wraps SlotFillingComponent so that it takes the slotfilling info from RASA configs"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save(self):
        pass

    def load(self, *args, **kwargs):
        """reads the slotfilling info from RASA-styled dataset"""
        domain_path = Path(self.load_path, MD_YAML_DialogsDatasetReader.DOMAIN_FNAME)
        nlu_path = Path(self.load_path, MD_YAML_DialogsDatasetReader.NLU_FNAME)
        domain_knowledge = DomainKnowledge(read_yaml(domain_path))
        # todo: rewrite MD_YAML_DialogsDatasetReader so that public methods are enough
        _, slot_name2text2value = MD_YAML_DialogsDatasetReader._read_intent2text_mapping(nlu_path, domain_knowledge)
        self._slot_vals = slot_name2text2value
