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

import math
from logging import getLogger
from typing import Optional

from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

log = getLogger(__name__)


class RepeatBatchGenerator:
    """Repeating dataset. If there is not enough elements in the dataset to form another batch, elements for the batch 
    are drawn in the beginning of the dataset. Optionally dataset is reshuffled before a repeat.

    Args:
        dataset_iterator: dataset iterator from which batches are drawn.
        batch_size: size fo the batch.
        data_type: "train", "valid", or "test"
        shuffle: whether dataset will be shuffled before each repeat.
        n_batches: the number of batches that will be generated.
        size_of_the_last_batch: used if dataset size is not evenly divisible by batch size.
    """
    def __init__(
            self, 
            dataset_iterator: DataLearningIterator,
            batch_size: int, 
            data_type: str, 
            shuffle: bool, 
            n_batches: Optional[int] = None, 
            size_of_last_batch: Optional[int] = None
    ):
        self.dataset_iterator = dataset_iterator
        self.batch_size = batch_size
        self.data_type = data_type
        self.shuffle = shuffle
        self.n_batches = n_batches
        self.size_of_last_batch = self.batch_size if size_of_last_batch is None else size_of_last_batch
        
        self.inner_batch_size = math.gcd(len(self.dataset_iterator.data[data_type]), batch_size)
        self.gen = self.dataset_iterator.gen_batches(self.inner_batch_size, self.data_type, self.shuffle)
        self.batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.n_batches is not None and self.batch_count > self.n_batches:
            raise StopIteration
        x, y = (), ()
        while len(x) < self.batch_size or len(y) < self.batch_size:
            try:
                xx, yy = next(self.gen)
            except StopIteration:
                self.gen = self.dataset_iterator.gen_batches(self.inner_batch_size, self.data_type, self.shuffle)
                continue
            assert len(xx) == self.inner_batch_size and len(yy) == self.inner_batch_size, \
                "self.inner_batch_size equals greatest common divisor of dataset size and " \
                "required batch size so dataset size has to divisible by task batch size evenly."
            x += xx
            y += yy
        assert len(x) == self.batch_size and len(y) == self.batch_size
        self.batch_count += 1
        if self.batch_count == self.n_batches:
            x = x[:self.size_of_last_batch]
            y = y[:self.size_of_last_batch]
        return x, y

