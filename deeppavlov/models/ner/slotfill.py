import json
from pathlib import Path

import tensorflow as tf
from fuzzywuzzy import process
from overrides import overrides
import pathlib

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.data.utils import download_untar, mark_done
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.models.ner.corpus import Corpus
from deeppavlov.models.ner.ner_network import NerNetwork
from deeppavlov.core.data.utils import tokenize_reg



@register('dstc_slotfilling')
class DstcSlotFillingNetwork(Inferable, Trainable):
    def __init__(self,
                 ner_network: NerNetwork,
                 # model_path,
                 **kwargs):
        model_path = pathlib.Path(self.model_path)
        # Check existance of the model files. Download model files if needed
        files_required = ['dict.txt', 'ner_model.ckpt.meta', 'params.json', 'slot_vals.json']
        for file_name in files_required:
            if not model_path.joinpath(file_name).exists():
                url = 'http://lnsigo.mipt.ru/export/ner_dstc_model.tar.gz'
                print('Loading model from {} to {}'.format(url, model_path))
                download_untar(url, model_path)
                mark_done(model_path)
                break

        slot_vals_filepath = model_path / 'slot_vals.json'

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._ner_network = ner_network
        with open(slot_vals_filepath) as f:
            self._slot_vals = json.load(f)

    # TODO: write load and save
    @overrides
    def load(self, *args, **kwargs):
        pass

    @overrides
    def save(self, *args, **kwargs):
        pass

    @overrides
    def train(self, data, num_epochs=10):
        for epoch in range(num_epochs):
            self._ner_network.train(data)

    @overrides
    def infer(self, instance, *args, **kwargs):
        if not len(instance.strip()):
            return dict()
        return self.predict_slots(instance)

    def interact(self):
        s = input('Type in the message you want to tag: ')
        prediction = self.predict_slots(s)
        print(prediction)

    def predict_slots(self, utterance):
        # For utterance extract named entities and perform normalization for slot filling
        tokens = tokenize_reg(utterance)
        with self.graph.as_default():
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
        # Example
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
