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

import random
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


def process_word(word, to_lower=False, append_case=None):
    if all(x.isupper() for x in word) and len(word) > 1:
        uppercase = "<ALL_UPPER>"
    elif word[0].isupper():
        uppercase = "<FIRST_UPPER>"
    else:
        uppercase = None
    if to_lower:
        word = word.lower()
    if word.isdigit():
        answer = ["<DIGIT>"]
    elif word.startswith("http://") or word.startswith("www."):
        answer = ["<HTTP>"]
    else:
        answer = list(word)
    if to_lower and uppercase is not None:
        if append_case == "first":
            answer = [uppercase] + answer
        elif append_case == "last":
            answer = answer + [uppercase]
    return tuple(answer)


def preprocess_data(data, to_lower=True, append_case="first"):
    new_data = []
    for words, tags in data:
        new_words = [process_word(word, to_lower=to_lower, append_case=append_case)
                     for word in words]
        # tags could also be processed in future
        new_tags = tags
        new_data.append((new_words, new_tags))
    return new_data


@register('morphotagger_dataset')
class MorphoTaggerDatasetIterator(DataLearningIterator):

    def __init__(self, data, seed=None, shuffle=True,
                 validation_split=0.2, bucket=True):

        # processed_data = {mode: preprocess_data(sample, to_lower=to_lower,
        #                                         append_case=append_case)
        #                   for mode, sample in data.items()}
        processed_data = data
        self.bucket = bucket
        self.validation_split = validation_split
        super().__init__(processed_data, seed, shuffle)

    def split(self):
        if len(self.valid) == 0:
            if self.shuffle:
                random.shuffle(self.train)
            L = int(len(self.train) * (1.0 - self.validation_split))
            self.train, self.valid = self.train[:L], self.valid[L:]
        return

    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None, return_indexes: bool = False):
        if shuffle is None:
            shuffle = self.shuffle
        data = self.data[data_type]
        if shuffle:
            random.shuffle(data)
        lengths = [len(x[0]) for x in data]
        indexes = np.argsort(lengths)
        L = len(data)
        if batch_size < 0:
            batch_size = L
        for start in range(0, L, batch_size):
            indexes_to_yield = indexes[start:start+batch_size]
            data_to_yield = tuple(zip(*([data[i] for i in indexes_to_yield])))
            if return_indexes:
                yield indexes_to_yield, data_to_yield
            else:
                yield data_to_yield