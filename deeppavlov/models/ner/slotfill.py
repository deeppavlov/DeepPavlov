import json

import tensorflow as tf
from fuzzywuzzy import process
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.models.ner.ner_network import NerNetwork
from deeppavlov.core.data.utils import tokenize_reg
from deeppavlov.core.data.utils import download
from deeppavlov.core.common.file import read_json


@register('dstc_slotfilling')
class DstcSlotFillingNetwork(Inferable, Trainable):
    def __init__(self, ner_network: NerNetwork,
                 save_path, load_path=None,
                 train_now=False, **kwargs):

        super().__init__(save_path=save_path, load_path=load_path,
                         train_now=train_now, mode=kwargs['mode'])

        # Check existance of file with slots, slot values, and corrupted (misspelled) slot values
        if not self.load_path.is_file():
            self.load()
            
        print("[ loading slot values from `{}` ]".format(str(self.load_path)))
        self._slot_vals = read_json(self.load_path)

        self._ner_network = ner_network
        self._ner_network.load()

    @overrides
    def save(self):
        self._ner_network.save()

    @overrides
    def train(self, data, num_epochs=2):
        if self.train_now:
            for epoch in range(num_epochs):
                self._ner_network.train(data)
                self._ner_network.eval_conll(data.iter_all('valid'), short_report=False,
                                             data_type='valid')
            self._ner_network.eval_conll(data.iter_all('train'), short_report=False,
                                         data_type='train')
            self._ner_network.eval_conll(data.iter_all('test'), short_report=False,
                                         data_type='test')
            self.save()
        else:
            self._ner_network.load()

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

    @overrides
    def load(self):
        url = 'http://lnsigo.mipt.ru/export/datasets/dstc_slot_vals.json'
        download(self.save_path, url)
