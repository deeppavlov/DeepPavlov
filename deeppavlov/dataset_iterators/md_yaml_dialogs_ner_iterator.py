# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import json
import os
import re
import tempfile
from logging import getLogger
from typing import Dict, List, Tuple, Any, Iterator

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.dataset_iterators.md_yaml_dialogs_iterator import \
    MD_YAML_DialogsDatasetIterator
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
from deeppavlov.dataset_readers.dto.rasa.domain_knowledge import DomainKnowledge
from deeppavlov.dataset_readers.dto.rasa.stories import Story, Turn, Stories
from deeppavlov.dataset_readers.dto.rasa.nlu import Intents

log = getLogger(__name__)


class RASADict(dict):
    def __add__(self, oth):
        return RASADict()




from typing import Dict, List, Tuple, Any, Iterator


@register('md_yaml_dialogs_ner_iterator')
class MD_YAML_DialogsDatasetNERIterator(MD_YAML_DialogsDatasetIterator):

    def __init__(self,
                 data: Dict[str, List[Tuple[Any, Any]]],
                 seed: int = None,
                 shuffle: bool = True,
                 limit: int = 10) -> None:
        super().__init__(data, seed, shuffle, limit)

    def gen_batches(self,
                    batch_size: int,
                    data_type: str = 'train',
                    shuffle: bool = None) -> Iterator[Tuple]:

        for batch in super().gen_batches(batch_size,
                                         data_type,
                                         shuffle):
            processed_data = list()
            processed_texts = dict()

            for xs, ys in zip(*batch):

                for x, y in zip(xs, ys):
                    text = x['text']
                    if not text.strip():
                        continue
                    intents = []
                    if 'intents' in x:
                        intents = x['intents']
                    elif 'slots' in x:
                        intents = [x]
                    # aggregate slots from different intents
                    slots = list()
                    for intent in intents:
                        current_slots = intent.get('slots', [])
                        for slot_type, slot_val in current_slots:
                            # if not self._slot_vals or (
                            #         slot_type in self._slot_vals):
                            slots.append((slot_type, slot_val,))
                    # remove duplicate pairs (text, slots)
                    if (text in processed_texts) and (
                            slots in processed_texts[text]):
                        continue
                    processed_texts[text] = processed_texts.get(text, []) + [
                        slots]
                    processed_data.append(self._add_bio_markup(text, slots))
            yield processed_data

    def _add_bio_markup(self,
                        utterance: str,
                        slots: List[Tuple[str, str]]) -> Tuple[List, List]:
        tokens = utterance.split()
        n_toks = len(tokens)
        tags = ['O' for _ in range(n_toks)]
        for n in range(n_toks):
            for slot_type, slot_val in slots:
                for entity in [slot_val]:
                    slot_tokens = entity.split()
                    slot_len = len(slot_tokens)
                    if n + slot_len <= n_toks and \
                            self._is_equal_sequences(tokens[n: n + slot_len],
                                                     slot_tokens):
                        tags[n] = 'B-' + slot_type
                        for k in range(1, slot_len):
                            tags[n + k] = 'I-' + slot_type
                        break
        return tokens, tags

    @staticmethod
    def _is_equal_sequences(seq1, seq2):
        equality_list = [tok1 == tok2 for tok1, tok2 in zip(seq1, seq2)]
        return all(equality_list)


@register("md_yaml_dialogs_intents_iterator")
class MD_YAML_DialogsDatasetIntentsIterator(MD_YAML_DialogsDatasetIterator):

    def __init__(self,
                 data: Dict[str, List[Tuple[Any, Any]]],
                 seed: int = None,
                 shuffle: bool = True,
                 limit: int = 10) -> None:
        super().__init__(data, seed, shuffle, limit)

    def gen_batches(self,
                    batch_size: int,
                    data_type: str = 'train',
                    shuffle: bool = None) -> Iterator[Tuple]:

        for batch in super().gen_batches(batch_size,
                                         data_type,
                                         shuffle):
            texts, intents = list(), list()
            for users, syss in zip(*batch):
                for user, sys in zip(users, syss):
                    reply = user
                    curr_intents = []
                    if reply['intents']:
                        for intent in reply['intents']:
                            for slot in intent['slots']:
                                if slot[0] == 'slot':
                                    curr_intents.append(
                                        intent['act'] + '_' + slot[1])
                                else:
                                    curr_intents.append(
                                        intent['act'] + '_' + slot[0])
                            if len(intent['slots']) == 0:
                                curr_intents.append(intent['act'])
                    else:
                        if reply['text']:
                            curr_intents.append('unknown')
                        else:
                            continue
                    texts.append(reply["text"])
                    intents.append(curr_intents)
                    # processed_data.append((reply['text'], curr_intents))
            yield texts, intents