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
from random import Random
from typing import Dict, Iterator, List, Tuple

from sklearn.model_selection import train_test_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


from deeppavlov.debug_helpers import recursive_shape  # FIXME: remove debug import


log = getLogger(__name__)


class RepeatBatchGenerator:
    def __init__(
            self, dataset_iterator, batch_size, data_type, shuffle, n_batches=float('inf'), size_of_last_batch=None):
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
        if self.batch_count > self.n_batches:
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


@register('multitask_iterator')
class MultiTaskIterator:
    """
    Class gets data dictionary from DatasetReader instance, merge fields if necessary, split a field if necessary

    Args:
        data: dictionary of data with fields "train", "valid" and "test" (or some of them)
        fields_to_merge: list of fields (out of ``"train", "valid", "test"``) to merge
        merged_field: name of field (out of ``"train", "valid", "test"``) to which save merged fields
        field_to_split: name of field (out of ``"train", "valid", "test"``) to split
        split_fields: list of fields (out of ``"train", "valid", "test"``) to which save splitted field
        split_proportions: list of corresponding proportions for splitting
        seed: random seed for iterating
        shuffle: whether to shuffle examples in batches
        split_seed: random seed for splitting dataset, if ``split_seed`` is None, division is based on `seed`.
        stratify: whether to use statified split
        *args: argument
        **kwargs: arguments

    Attributes:
        data: dictionary of data with fields "train", "valid" and "test" (or some of them)
    """

    def __init__(self, data: dict, tasks: dict):
        """
        Initialize dataset using data from DatasetReader,
        merges and splits fields according to the given parameters.
        """
        self.data = data
        self.task_iterators_params = tasks
        self.task_iterators = {}
        for task_name, task_iterator_params in self.task_iterators_params.items():
            task_iterator_params = copy.deepcopy(task_iterator_params)
            task_iterator_params['class_name'] = task_iterator_params['iterator_class_name']
            del task_iterator_params['iterator_class_name']
            self.task_iterators[task_name] = from_params(task_iterator_params, data=data[task_name])
            log.debug(f"(MultitaskIterator.__init__)self.task_iterators[{task_name}]: {self.task_iterators[task_name]}")
            log.debug(f"(MultitaskIterator.__init__)self.task_iterators['{task_name}'].train length: {len(self.task_iterators[task_name].train)}")
            log.debug(f"(MultitaskIterator.__init__)self.task_iterators['{task_name}'].valid length: {len(self.task_iterators[task_name].valid)}")
            log.debug(f"(MultitaskIterator.__init__)self.task_iterators['{task_name}'].test length: {len(self.task_iterators[task_name].test)}")

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
        # TODO: write detailed commentaries for this method
        log.debug(f"(MultitaskIterator.gen_batches)batch_size data_type: {batch_size} {data_type}")
        max_task_data_len = max([len(iter_.data[data_type]) for iter_ in self.task_iterators.values()])
        size_of_last_batch = max_task_data_len % batch_size
        n_batches = max_task_data_len // batch_size
        for task_batches in zip(
                *[RepeatBatchGenerator(iter_, batch_size, data_type, shuffle, n_batches, size_of_last_batch) for 
                  iter_ in self.task_iterators.values()]
        ):
            x_instances, y_instances = [], []
            for task_batch in task_batches:
                x_instances.append(task_batch[0])
                y_instances.append(task_batch[1])
            b = (tuple(zip(*x_instances)), tuple(zip(*y_instances)))
            log.debug(f"(MultitaskIterator.gen_batches)batch shape: {recursive_shape(b)}")
            yield b

    def get_instances(self, data_type: str = 'train'):
        max_task_data_len = max([len(iter_.get_instances()[0]) for iter_ in self.task_iterators.values()])
        log.debug(f"(MultitaskIterator.get_instances)max_task_data_len: {max_task_data_len}")
        x_instances = []
        y_instances = []
        for task_name, iter_ in self.task_iterators.items():
            x, y = iter_.get_instances()
            log.debug(f"(MultitaskIterator.get_instances)len(x) for {task_name}: {len(x)}")
            n_repeats = math.ceil(max_task_data_len / len(x))
            x *= n_repeats
            y *= n_repeats
            x_instances.append(x[:max_task_data_len])
            y_instances.append(y[:max_task_data_len])
            
        instances = (tuple(zip(*x_instances)), tuple(zip(*y_instances)))
        log.debug(f"(MultitaskIterator.get_instances)instances.shape: {recursive_shape(instances)}")
        return instances
