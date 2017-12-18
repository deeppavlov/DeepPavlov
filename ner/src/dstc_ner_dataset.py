from deeppavlov.data.dataset import Dataset
import json
from deeppavlov.data.utils import download

class DstcNerDataset(Dataset):

    def __init__(self, data):
        r""" Dataset takes a dict with fields 'train', 'test', 'valid'. A list of samples (pairs x, y) is stored
             in each field.

             Args:
                data: list of (x, y) pairs. Each pair is a sample from the dataset. x as well as y can be a tuple
                    of different input features.
        """

        # TODO: add external building
        with open('slot_vals.json') as f:
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

    def _preprocess(self, data_part):
        processed_data_part = list()
        for sample in data_part:
            for utterance in sample:
                text = utterance['context']['text']
                intents = utterance['context'].get('intents', dict())
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
                    if n + slot_len < n_toks and self._is_equal_sequences(tokens[n: n + slot_len], slot_tokens):
                        tags[n] = 'B-' + slot_type
                        for k in range(1, slot_len):
                            tags[n + k] = 'I-' + slot_type
                        break
        return tokens, tags

    def _is_equal_sequences(self, seq1, seq2):
        equality_list = [tok1 == tok2 for tok1, tok2 in zip(seq1, seq2)]
        return all(equality_list)

    def _build_slot_vals(self, slot_vals_json_path='data/'):
        url = 'http://lnsigo.mipt.ru/export/datasets/dstc_slot_vals.json'
        download(slot_vals_json_path, url)
