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

import random
from typing import List, Dict, Generator, Tuple, Any
import numpy as np
from sklearn.model_selection import train_test_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset
from deeppavlov.models.embedders.fasttext_embedder import EmbeddingsDict
from deeppavlov.models.intent_recognition.intent_cnn_keras.intent_model import KerasIntentModel
from deeppavlov.models.intent_recognition.intent_cnn_keras.utils import labels2onehot, proba2labels, proba2onehot

@register('intent_dataset')
class IntentDataset(Dataset):

    @staticmethod
    def texts2vec(sentences, embedding_dict, text_size, embedding_size):
        embeddings_batch = []
        for sen in sentences:
            embeddings = []
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            if len(tokens) > text_size:
                tokens = tokens[:text_size]
            for tok in tokens:
                embeddings.append(embedding_dict.tok2emb.get(tok))
            if len(tokens) < text_size:
                pads = [np.zeros(embedding_size)
                        for _ in range(text_size - len(tokens))]
                embeddings = pads + embeddings
            embeddings = np.asarray(embeddings)
            embeddings_batch.append(embeddings)
        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

    def embedded_batch_generator(self, embedding_dict, text_size: int, embedding_size: int, classes,
                                 batch_size: int, data_type: str = 'train') -> Generator:
        r"""This function returns a generator, which serves for generation of raw (no preprocessing such as tokenization)
         batches

        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'

        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
        """
        data = self.data[data_type]
        data_len = len(data)
        order = list(range(data_len))

        rs = random.getstate()
        random.setstate(self.random_state)
        random.shuffle(order)
        self.random_state = random.getstate()
        random.setstate(rs)

        for i in range((data_len - 1) // batch_size + 1):
            batch = list(zip(*[data[o] for o in order[i*batch_size:(i+1)*batch_size]]))
            embedding_dict.add_items(batch[0])

            batch[0] = IntentDataset.texts2vec(batch[0], embedding_dict, text_size, embedding_size)
            batch[1] = labels2onehot(batch[1], classes=classes)
            yield batch

    def extract_classes(self, *args, **kwargs):
        intents = []
        all_data = self.iter_all(data_type='train')
        for sample in all_data:
            intents.extend(sample[1])
        if 'valid' in self.data.keys():
            all_data = self.iter_all(data_type='valid')
            for sample in all_data:
                intents.extend(sample[1])
        intents = np.unique(intents)
        return np.array(sorted(intents))

    def split_data(self, field_to_split, new_fields, proportions):
        data_to_div = self.data[field_to_split].copy()
        data_size = len(self.data[field_to_split])
        for i in range(len(new_fields) - 1):
            self.data[new_fields[i]], data_to_div = train_test_split(data_to_div,
                                                                     test_size=len(data_to_div) -
                                                                               int(data_size * proportions[i]))
        self.data[new_fields[-1]] = data_to_div
        return True

    def merge_data(self, fields_to_merge, new_field):
        data = self.data.copy()
        data[new_field] = []
        for name in fields_to_merge:
            data[new_field] += self.data[name]
        self.data = data
        return True
