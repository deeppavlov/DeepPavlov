import json
import glob

import tensorflow as tf
from fuzzywuzzy import process
from overrides import overrides
import pathlib

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.models.ner.ner_network import NerNetwork
from deeppavlov.core.data.utils import tokenize_reg
from deeppavlov.core.data.utils import download



@register('dstc_slotfilling')
class DstcSlotFillingNetwork(Inferable, Trainable):
    def __init__(self,
                 ner_network: NerNetwork):
        # Make it path
        self.model_path = pathlib.Path(self.model_path)

        # Check existance of file with slots, slot values, and corrupted (misspelled) slot values
        slot_vals_filepath = self.model_path / 'slot_vals.json'
        if not slot_vals_filepath.is_file():
            self._download_slot_vals(slot_vals_filepath)
        self._ner_model_path = self.model_path / 'dstc_ner_network.ckpt'

        self._ner_network = ner_network
        self.load()
        with open(slot_vals_filepath) as f:
            self._slot_vals = json.load(f)

    @overrides
    def load(self):
        # Check prescence of the model files
        path = str(self.model_path.absolute())
        if tf.train.get_checkpoint_state(path) is not None:
            print('Loading model from {}'.format(path))
            self._ner_network.load(self._ner_model_path)
        # else:
        #     raise Warning('Error while loading NER model. There must be 3 dstc_ner_network.ckpt files!')

    @overrides
    def save(self):
        self._ner_network.save(self._ner_model_path)

    @overrides
    def train(self, data, num_epochs=3):
        for epoch in range(num_epochs):
            self._ner_network.train(data)
            self._ner_network.eval_conll(data.iter_all('valid'), short_report=False, data_type='valid')
        self._ner_network.eval_conll(data.iter_all('train'), short_report=False, data_type='train')
        self._ner_network.eval_conll(data.iter_all('valid'), short_report=False, data_type='valid')
        self._ner_network.eval_conll(data.iter_all('test'), short_report=False, data_type='test')
        self.save()

    @overrides
    def infer(self, instance, *args, **kwargs):
        instance = instance.strip()
        if not len(instance):
            return dict()
        return self.predict_slots(instance.lower())

    def interact(self):
        s = input('Type in the message you want to tag: ')
        prediction = self.predict_slots(s)
        print(prediction)

    def predict_slots(self, utterance):
        # For utterance extract named entities and perform normalization for slot filling
        tokens = tokenize_reg(utterance)
        tags = self._ner_network.predict_for_token_batch([tokens])[0]
        entities, slots = self._chunk_finder(tokens, tags)
        slot_values = dict()
        for entity, slot in zip(entities, slots):
            slot_values[slot] = self.ner2slot(entity, slot)
        return slot_values

    def ner2slot(self, input_entity, slot):
        # Given named entity return normalized slot value
        if isinstance(input_entity, list):
            input_entity = ' '.join(input_entity)
        entities = list()
        normalized_slot_vals = list()
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
        chunk_tokens = list()
        entities = list()
        slots = list()
        for token, tag in zip(tokens, tags):
            curent_tag = tag.split('-')[-1].strip()
            current_prefix = tag.split('-')[0]
            if tag.startswith('B-'):
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = list()
                chunk_tokens.append(token)
            if current_prefix == 'I':
                if curent_tag != prev_tag:
                    if len(chunk_tokens) > 0:
                        entities.append(' '.join(chunk_tokens))
                        slots.append(prev_tag)
                        chunk_tokens = list()
                else:
                    chunk_tokens.append(token)
            if current_prefix == 'O':
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = list()
            prev_tag = curent_tag
        if len(chunk_tokens) > 0:
            entities.append(' '.join(chunk_tokens))
            slots.append(prev_tag)
        return entities, slots

    def shutdown(self):
        with self.graph.as_default():
            self.ner_network.shutdown()

    def reset(self):
        pass

    @staticmethod
    def _download_slot_vals(slot_vals_json_path):
        url = 'http://lnsigo.mipt.ru/export/datasets/dstc_slot_vals.json'
        download(slot_vals_json_path, url)
