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
import inspect

import tensorflow as tf
from fuzzywuzzy import process
from overrides import overrides
from copy import deepcopy
from pathlib import Path

from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.models.ner.network import NerNetwork
from deeppavlov.core.data.utils import tokenize_reg, download, download_decompress
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('dstc_slotfilling')
class DstcSlotFillingNetwork(TFModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        opt = deepcopy(kwargs)
        vocabs = opt.pop('vocabs')
        opt.update(vocabs)
        self.opt = opt

        # Find all input parameters of the network init
        network_parameter_names = list(inspect.signature(NerNetwork.__init__).parameters)

        # Fill all provided parameters from opt
        network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        # Initialize the network
        self.sess = tf.Session()
        network_parameters['sess'] = self.sess
        self._ner_network = NerNetwork(**network_parameters)

        download_best_model = opt.get('download_best_model', False)
        if download_best_model:
            model_path = str(self.load_path.parent.absolute())
            best_model_url = 'http://lnsigo.mipt.ru/export/models/ner/ner_dstc_model.tar.gz'
            download_decompress(best_model_url, model_path)

        # Training parameters
        # Find all parameters for network train
        train_parameters_names = list(inspect.signature(NerNetwork.train_on_batch).parameters)
        train_parameters = {par: opt[par] for par in train_parameters_names if par in opt}
        self.train_parameters = train_parameters

        # Check existance of file with slots, slot values, and corrupted (misspelled) slot values
        slot_vals_filepath = Path(self.save_path.parent) / 'slot_vals.json'
        if not slot_vals_filepath.is_file():
            self._download_slot_vals()

        with open(slot_vals_filepath) as f:
            self._slot_vals = json.load(f)

        if self.load_path is not None:
            self.load()

    @check_attr_true('train_now')
    def train_on_batch(self, batch_x, batch_y):
        self._ner_network.train_on_batch(batch_x, batch_y, **self.train_parameters)

    @overrides
    def __call__(self, batch, *args, **kwargs):
        if isinstance(batch[0], str):
            batch = [tokenize_reg(instance.strip()) for instance in batch]

        slots = [{}] * len(batch)

        m = [i for i, v in enumerate(batch) if v]
        if m:
            batch = [batch[i] for i in m]
            tags_batch = self._ner_network.predict_for_token_batch(batch)
            for i, tokens, tags in zip(m, batch, tags_batch):
                slots[i] = self.predict_slots(tokens, tags)
        return slots

    def predict_slots(self, tokens, tags):
        # For utterance extract named entities and perform normalization for slot filling

        entities, slots = self._chunk_finder(tokens, tags)
        slot_values = {}
        for entity, slot in zip(entities, slots):
            slot_values[slot] = self.ner2slot(entity, slot)
        return slot_values

    def ner2slot(self, input_entity, slot):
        # Given named entity return normalized slot value
        if isinstance(input_entity, list):
            input_entity = ' '.join(input_entity)
        entities = []
        normalized_slot_vals = []
        for entity_name in self._slot_vals[slot]:
            for entity in self._slot_vals[slot][entity_name]:
                entities.append(entity)
                normalized_slot_vals.append(entity_name)
        best_match = process.extract(input_entity, entities, limit=2 ** 20)[0][0]
        return normalized_slot_vals[entities.index(best_match)]

    @staticmethod
    def _chunk_finder(tokens, tags):
        # For BIO labeled sequence of tags extract all named entities form tokens
        prev_tag = ''
        chunk_tokens = []
        entities = []
        slots = []
        for token, tag in zip(tokens, tags):
            curent_tag = tag.split('-')[-1].strip()
            current_prefix = tag.split('-')[0]
            if tag.startswith('B-'):
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = []
                chunk_tokens.append(token)
            if current_prefix == 'I':
                if curent_tag != prev_tag:
                    if len(chunk_tokens) > 0:
                        entities.append(' '.join(chunk_tokens))
                        slots.append(prev_tag)
                        chunk_tokens = []
                else:
                    chunk_tokens.append(token)
            if current_prefix == 'O':
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = []
            prev_tag = curent_tag
        if len(chunk_tokens) > 0:
            entities.append(' '.join(chunk_tokens))
            slots.append(prev_tag)
        return entities, slots

    def shutdown(self):
        with self.graph.as_default():
            self.ner_network.shutdown()

    def _download_slot_vals(self):
        url = 'http://lnsigo.mipt.ru/export/datasets/dstc_slot_vals.json'
        download(self.save_path.parent / 'slot_vals.json', url)
