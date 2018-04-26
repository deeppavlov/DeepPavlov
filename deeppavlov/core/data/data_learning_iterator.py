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

from random import Random
from typing import List, Dict, Generator, Tuple, Any

from deeppavlov.core.common.registry import register


@register('data_learning_iterator')
class DataLearningIterator:
    """
    Dataset iterator for learning models, e. g. neural networks. Split data on 'train', 'valid',
    'test' sets. Generate batches.
    """
    def split(self, *args, **kwargs):
        pass

    def __init__(self, data: Dict[str, List[Tuple[Any, Any]]],
                 seed: int = None, shuffle: bool = True,
                 *args, **kwargs) -> None:
        """ Dataiterator takes a dict with fields 'train', 'test', 'valid'. A list of samples
         (pairs x, y) is stored in each field.
        Args:
            data: list of (x, y) pairs. Each pair is a sample from the dataset. x as well as y
            can be a tuple of different input features.
            seed (int): random seed for data shuffling. Defaults to None
            shuffle: whether to shuffle data when batching (from config)
        """
        self.shuffle = shuffle

        self.random = Random(seed)

        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None) -> Generator:
        """Return a generator, which serves for generation of raw (no preprocessing such as tokenization)
        batches
        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'
            shuffle (bool): whether to shuffle dataset before batching
        Returns:
            batch_gen (Generator): a generator, that iterates through the part (defined by data_type) of the dataset
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

    def get_instances(self, data_type: str = 'train') -> tuple:
        """
        Reformat data to x, y pairs, where x, y is a single dataset instance.
        Args:
            data_type (str): can be either 'train', 'test', or 'valid'
        Returns:
            x, y pairs
        """
        data = self.data[data_type]
        return tuple(zip(*data))
