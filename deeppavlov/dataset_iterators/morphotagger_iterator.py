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

from typing import Tuple, List, Dict, Any, Iterator

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('morphotagger_dataset_iterator')
class MorphoTaggerDatasetIterator(DataLearningIterator):
    """
    Iterates over data for Morphological Tagging.
    A subclass of :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`.

    Args:
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching
        validation_split: the fraction of validation data
            (is used only if there is no `valid` subset in `data`)
    """

    def __init__(self, data: Dict[str, List[Tuple[Any, Any]]], seed: int = None,
                 shuffle: bool = True, validation_split: float = 0.2) -> None:
        self.validation_split = validation_split
        super().__init__(data, seed, shuffle)

    def split(self, *args, **kwargs) -> None:
        """
        Splits the `train` part to `train` and `valid`, if no `valid` part is specified.
        Moves deficient data from `valid` to `train` if both parts are given,
        but `train` subset is too small.
        """
        if len(self.valid) == 0:
            if self.shuffle:
                self.random.shuffle(self.train)
            L = int(len(self.train) * (1.0 - self.validation_split))
            self.train, self.valid = self.train[:L], self.train[L:]

    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None, return_indexes: bool = False) -> Iterator[tuple]:
        """Generate batches of inputs and expected output to train neural networks
        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching
            return_indexes: whether to return indexes of batch elements in initial dataset
        Yields:
            a tuple of a batch of inputs and a batch of expected outputs.
            If `return_indexes` is True, also yields indexes of batch elements.
        """
        if shuffle is None:
            shuffle = self.shuffle
        data = self.data[data_type]
        lengths = [len(x[0]) for x in data]
        indexes = np.argsort(lengths)
        L = len(data)
        if batch_size < 0:
            batch_size = L
        starts = list(range(0, L, batch_size))
        if shuffle:
            self.random.shuffle(starts)
        for start in starts:
            indexes_to_yield = indexes[start:start + batch_size]
            data_to_yield = tuple(list(x) for x in zip(*([data[i] for i in indexes_to_yield])))
            if return_indexes:
                yield indexes_to_yield, data_to_yield
            else:
                yield data_to_yield
