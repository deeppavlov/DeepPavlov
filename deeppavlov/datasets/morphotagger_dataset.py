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
from deeppavlov.core.data.dataset import Dataset


@register('morphotagger_dataset')
class MorphoTaggerDataset(Dataset):

    def __init__(self, data, seed=None, shuffle=True,
                 validation_split=0.2, bucket=True):
        self.bucket = bucket
        self.validation_split = validation_split
        super().__init__(data, seed, shuffle)

    def split(self):
        if len(self.valid) == 0:
            if self.shuffle:
                random.shuffle(self.train)
            L = int(len(self.train) * (1.0 - self.validation_split))
            self.train, self.valid = self.train[:L], self.valid[L:]
        return

    def batch_generator(self, batch_size: int, data_type: str = 'train',
                        shuffle: bool = None):
        if shuffle is None:
            shuffle = self.shuffle
        data = self.data[data_type]
        if shuffle:
            random.shuffle(data)
        lengths = [len(x[0]) for x in data]
        indexes = np.argsort(lengths)
        L = len(data)
        for start in range(0, L, batch_size):
            yield tuple(zip(*([data[i] for i in indexes[start:start+batch_size]])))