from .src.corpus import Corpus
from .src.ner_network import NerNetwork
from fuzzywuzzy import process
import json
from deeppavlov.common.registry import register_model
from .utils.nlputils import tokenize
from deeppavlov.models.inferable import Inferable
from overrides import overrides


@register_model('dstc_slotfilling')
class DstcSlotFillingNetwork(Inferable):
    def __init__(self,
                 dict_filepath='model/dict.txt',
                 model_filepath='model/ner_model.ckpt',
                 params_filepath='model/params.json',
                 slot_vals_filepath='model/slot_vals.json'):

        # Build and initialize the model
        with open(params_filepath) as f:
            network_params = json.load(f)
        self._corpus = Corpus(dicts_filepath=dict_filepath)
        self._ner_network = NerNetwork(self._corpus, pretrained_model_filepath=model_filepath, **network_params)
        with open(slot_vals_filepath) as f:
            self._slot_vals = json.load(f)

    @overrides
    def infer(self, instance, *args, **kwargs):
        if len(instance.split()):
            return dict()
        return self.predict_slots(instance)

    def interact(self):
        s = input('Type in the message you want to tag: ')
        prediction = self.predict_slots(s)
        print(prediction)

    def predict_slots(self, utterance):
        # For utterance extract named entities and perform normalization for slot filling
        tokens = tokenize(utterance)
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
            curent_tag = tag.split('-')[-1]
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

        return entities, slots

    def reset(self):
        pass
