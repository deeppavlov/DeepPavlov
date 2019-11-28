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

from random import Random
from typing import List, Dict, Tuple, Any, Iterator

from deeppavlov.core.common.registry import register


@register('data_learning_iterator')
class DataLearningIterator:
    """Dataset iterator for learning models, e. g. neural networks.

    Args:
        data: list of (x, y) pairs for every data type in ``'train'``, ``'valid'`` and ``'test'``
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching

    Attributes:
        shuffle: whether to shuffle data during batching
        random: instance of ``Random`` initialized with a seed
    """

    def split(self, *args, **kwargs):
        """ Manipulate self.train, self.valid, and self.test into their final form. """
        pass

    def preprocess(self, data: List[Tuple[Any, Any]], *args, **kwargs) -> List[Tuple[Any, Any]]:
        """ Transform the data for a specific data type (e.g. ``'train'``). """
        return data

    def __init__(self, data: Dict[str, List[Tuple[Any, Any]]], seed: int = None, shuffle: bool = True,
                 *args, **kwargs) -> None:
        self.shuffle = shuffle

        self.random = Random(seed)

        self.train = self.preprocess(data.get('train', []), *args, **kwargs)
        self.valid = self.preprocess(data.get('valid', []), *args, **kwargs)
        self.test = self.preprocess(data.get('test', []), *args, **kwargs)
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None) -> Iterator[Tuple[tuple, tuple]]:
        """Generate batches of inputs and expected output to train neural networks

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching

        Yields:
             a tuple of a batch of inputs and a batch of expected outputs
        """
        if shuffle is None:
            shuffle = self.shuffle

        data = self.data[data_type]
        data_len = len(data)

        if data_len == 0:
            return

        order = list(range(data_len))
        if shuffle:
            self.random.shuffle(order)

        if batch_size < 0:
            batch_size = data_len

        for i in range((data_len - 1) // batch_size + 1):
            yield tuple(zip(*[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]))

    def get_instances(self, data_type: str = 'train') -> Tuple[tuple, tuple]:
        """Get all data for a selected data type

        Args:
            data_type (str): can be either ``'train'``, ``'test'``, ``'valid'`` or ``'all'``

        Returns:
             a tuple of all inputs for a data type and all expected outputs for a data type
        """
        data = self.data[data_type]
        return tuple(zip(*data))
