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

import copy
import math
from logging import getLogger
from typing import Iterator, Optional, Tuple, Union

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

log = getLogger(__name__)


@register('multitask_iterator')
class MultiTaskIterator:
    """
    Class merges data from several dataset iterators. When used for batch generation batches from
    merged dataset iterators are united into one batch. If sizes of merged datasets are different
    smaller datasets are repeated until their size becomes equal to the largest dataset.

    Args:
        data: dictionary which keys are task names and values are dictionaries with fields
            ``"train", "valid", "test"``.
        tasks: dictionary which keys are task names and values are init params of dataset iterators.

    Attributes:
        data: dictionary of data with fields "train", "valid" and "test" (or some of them)
    """

    def __init__(self, data: dict, tasks: dict):
        self.task_iterators = {}
        for task_name, task_iterator_params in tasks.items():
            task_iterator_params = copy.deepcopy(task_iterator_params)
            task_iterator_params['class_name'] = task_iterator_params['iterator_class_name']
            del task_iterator_params['iterator_class_name']
            self.task_iterators[task_name] = from_params(task_iterator_params, data=data[task_name])

        self.train = self._extract_data_type('train')
        self.valid = self._extract_data_type('valid')
        self.test = self._extract_data_type('test')
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self._unite_dataset_parts(self.train, self.valid, self.test)
        }

    def _extract_data_type(self, data_type):
        dataset_part = {}
        for task, iterator in self.task_iterators.items():
            dataset_part[task] = getattr(iterator, data_type)
        return dataset_part

    @staticmethod
    def _unite_dataset_parts(*dataset_parts):
        united = {}
        for ds_part in dataset_parts:
            for task, data in ds_part.items():
                if task not in united:
                    united[task] = data
                else:
                    united[task] = united[task] + data
        return united

    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None) -> Iterator[Tuple[tuple, tuple]]:
        """Generate batches and expected output to train neural networks. Batches from task iterators
        are united into one batch. Every element of the largest dataset is used once whereas smaller
        datasets are repeated until their size is equal to the largest dataset.

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching

        Yields:
            a tuple of a batch of inputs and a batch of expected outputs. Inputs and outputs are tuples.
            Element of inputs or outputs is a tuple which elements are x values of merged tasks in the order
            tasks are present in `tasks` argument of `__init__` method.
        """
        max_task_data_len = max([len(iter_.data[data_type]) for iter_ in self.task_iterators.values()])

        size_of_last_batch = max_task_data_len % batch_size
        if size_of_last_batch == 0:
            size_of_last_batch = batch_size

        n_batches = math.ceil(max_task_data_len / batch_size)
        for task_batches in zip(
                *[RepeatBatchGenerator(iter_, batch_size, data_type, shuffle, n_batches, size_of_last_batch) for 
                  iter_ in self.task_iterators.values()]
        ):
            x_instances, y_instances = [], []
            for task_batch in task_batches:
                x_instances.append(task_batch[0])
                y_instances.append(task_batch[1])
            b = (tuple(zip(*x_instances)), tuple(zip(*y_instances)))
            yield b

    def get_instances(self, data_type: str = 'train'):
        """Returns a tuple of inputs and outputs from all datasets. Lengths of inputs and outputs are equal to
        the size of the largest dataset. Smaller datasets are repeated until their sizes are equal to the
        size of the largest dataset.

        Args:
            data_type: can be either 'train', 'test', or 'valid'

        Returns:
            a tuple of all inputs for a data type and all expected outputs for a data type
        """
        max_task_data_len = max(
            [len(iter_.get_instances(data_type)[0]) for iter_ in self.task_iterators.values()])
        x_instances = []
        y_instances = []
        for task_name, iter_ in self.task_iterators.items():
            x, y = iter_.get_instances(data_type)
            n_repeats = math.ceil(max_task_data_len / len(x))
            x *= n_repeats
            y *= n_repeats
            x_instances.append(x[:max_task_data_len])
            y_instances.append(y[:max_task_data_len])
            
        instances = (tuple(zip(*x_instances)), tuple(zip(*y_instances)))
        return instances


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
            dataset_iterator: Union[MultiTaskIterator, DataLearningIterator], 
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

