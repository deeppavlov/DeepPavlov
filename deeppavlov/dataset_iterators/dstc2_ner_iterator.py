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

import logging
import json
from random import Random

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.data.utils import download

logger = logging.getLogger(__name__)


@register('dstc2_ner_iterator')
class Dstc2NerDatasetIterator(DataLearningIterator):

    def __init__(self, data, dataset_path, seed=None, shuffle=False):
        r""" Dataset takes a dict with fields 'train', 'test', 'valid'. A list of samples (pairs x, y) is stored
             in each field.

             Args:
                data: list of (x, y) pairs. Each pair is a sample from the dataset. x as well as y can be a tuple
                    of different input features.
        """
        self.shuffle = shuffle
        self.random = Random(seed)
        # TODO: include slot vals to dstc2.tar.gz
        dataset_path = expand_path(dataset_path) / 'slot_vals.json'
        self._build_slot_vals(dataset_path)
        with open(dataset_path, encoding='utf8') as f:
            self._slot_vals = json.load(f)
        for data_type in ['train', 'test', 'valid']:
            bio_markup_data = self._preprocess(data.get(data_type, []))
            setattr(self, data_type, bio_markup_data)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }
        self.shuffle = shuffle

    def _preprocess(self, data_part):
        processed_data_part = list()
        for sample in data_part:
            for utterance in sample:
                if 'intents' not in utterance or len(utterance['text']) < 1:
                    continue
                text = utterance['text']
                intents = utterance.get('intents', dict())
                slots = list()
                for intent in intents:

                    current_slots = intent.get('slots', [])
                    for slot_type, slot_val in current_slots:
                        if slot_type in self._slot_vals:
                            slots.append((slot_type, slot_val,))

                processed_data_part.append(self._add_bio_markup(text, slots))
        return processed_data_part

    def _add_bio_markup(self, utterance, slots):
        tokens = utterance.split()
        n_toks = len(tokens)
        tags = ['O' for _ in range(n_toks)]
        for n in range(n_toks):
            for slot_type, slot_val in slots:
                for entity in self._slot_vals[slot_type][slot_val]:
                    slot_tokens = entity.split()
                    slot_len = len(slot_tokens)
                    if n + slot_len <= n_toks and self._is_equal_sequences(tokens[n: n + slot_len],
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

    @staticmethod
    def _build_slot_vals(slot_vals_json_path='data/'):
        url = 'http://lnsigo.mipt.ru/export/datasets/dstc_slot_vals.json'
        download(slot_vals_json_path, url)
