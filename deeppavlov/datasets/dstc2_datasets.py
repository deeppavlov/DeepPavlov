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
import itertools
from overrides import overrides
from typing import Dict, Tuple, List, Generator, Any
import json
import random
import pathlib

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset
from deeppavlov.core.data.utils import download

logger = logging.getLogger(__name__)


@register('dstc2_dialog_dataset')
class DSTC2DialogDataset(Dataset):
    @overrides
    def __init__(self, data: Dict[str, List[Tuple[Any, Any]]], *args, **kwargs) \
            -> None:
        def _wrap(turn):
            x = turn[0]['text']
            y = turn[1]['text']
            other = {}
            other['act'] = turn[1]['act']
            if turn[0].get('db_result') is not None:
                other['db_result'] = turn[0]['db_result']
            if turn[0].get('episode_done'):
                other['episode_done'] = True
            return (x, y, other)

        self.train = list(map(_wrap, data.get('train', [])))
        self.valid = list(map(_wrap, data.get('valid', [])))
        self.test = list(map(_wrap, data.get('test', [])))
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

    @overrides
    def batch_generator(self, batch_size: int, data_type: str = 'train',
                        shuffle: bool = True) -> Generator:
        def _dialog(idx):
            return data[idx['start']: idx['end']]

        data = self.data[data_type]
        dialog_indices = self._dialog_indices(data)
        num_dialogs = len(dialog_indices)
        order = list(range(num_dialogs))
        if shuffle:
            random.shuffle(order)
        for i in range((num_dialogs - 1) // batch_size + 1):
            print("Getting dialogs =", [dialog_indices[o] for o in
                                        order[i * batch_size:(i + 1) * batch_size]])
            yield list(itertools.chain.from_iterable(
                _dialog(dialog_indices[o]) \
                for o in order[i * batch_size:(i + 1) * batch_size]))

    @staticmethod
    def _dialog_indices(data):
        dialog_indices = []
        i, last_idx = 0, 0
        dialog = {}
        for turn in data:
            if turn[2].get('episode_done'):
                if dialog:
                    dialog['end'] = i
                    last_idx = i
                    dialog_indices.append(dialog)
                dialog = {'start': last_idx}
            i += 1
        dialog['end'] = i
        dialog_indices.append(dialog)
        return dialog_indices

    # @staticmethod
    # def save_vocab(turns, fpath):
    #     print("Saving data to `{}`".format(fpath))
    #     with open(fpath, 'wt') as f:
    #         words = sorted(set(itertools.chain.from_iterable(
    #             turn[0].lower().split() for turn in turns
    #         )))
    #         f.write(' '.join(words))

    @overrides
    def iter_all(self, data_type: str = 'train') -> Generator:
        data = self.data[data_type]
        for instance in data:
            yield instance


@register('dstc2_ner_dataset')
class DstcNerDataset(Dataset):
    def __init__(self, data, dataset_path, shuffle=True, seed=None):
        r""" Dataset takes a dict with fields 'train', 'test', 'valid'. A list of samples (pairs x, y) is stored
             in each field.

             Args:
                data: list of (x, y) pairs. Each pair is a sample from the dataset. x as well as y can be a tuple
                    of different input features.
        """
        self.random_state = random.getstate()
        dataset_path = pathlib.Path(dataset_path) / 'slot_vals.json'
        self._build_slot_vals(dataset_path)
        with open(dataset_path) as f:
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
        self.seed = None

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
