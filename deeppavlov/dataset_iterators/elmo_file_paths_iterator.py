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


from typing import List, Tuple, Iterator, Optional
from  collections import deque

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

@register('elmo_file_paths_iterator')
class ELMoFilePathsIterator(DataLearningIterator):
    """Dataset iterator for tokenized datasetes like 1 Billion Word Benchmark

    Args:
        data: dict with keys ``'train'``, ``'valid'`` and ``'test'`` and values
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching

    Attributes:
        shuffle: whether to shuffle data during batching
        random: instance of ``Random`` initialized with a seed
    # """

    def __init__(self, 
                 data: dict, 
                 seed: Optional[int] = None, 
                 shuffle: bool = True,
                 unroll_steps: Optional[int] = None, 
                 n_gpus: Optional[int] = None, 
                 *args, **kwargs) -> None:
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self.unroll_steps = unroll_steps
        self.n_gpus = n_gpus
        super().__init__(data, seed, shuffle, *args, **kwargs)

    @staticmethod
    def _chunk_generator(items_list, chunk_size):
        for i in range(0, len(items_list), chunk_size):
            yield items_list[i:i + chunk_size]


    @staticmethod
    def _shard_generator(shards, shuffle = False, random = None):
        shards_to_choose = list(shards)
        if shuffle:
            random.shuffle(shards_to_choose)
        for shard in shards_to_choose:
            log.info(f'Loaded shard from {shard}')
            lines = open(shard).readlines()
            if shuffle:
                random.shuffle(lines)
            yield lines
            
    def _line_generator(self, shard_generator):
        for shard in shard_generator:
            line_generator = self._chunk_generator(shard, 1)
            for line in line_generator:
                yield line[0]

    @staticmethod
    def _batch_generator(line_generator, batch_size, unroll_steps):
        batch = [[] for i in range(batch_size)]
        stream = [[] for i in range(batch_size)]

        try:
            while True:
                for batch_item, stream_item in zip(batch, stream):
                    while len(stream_item) < unroll_steps: 
                        line = next(line_generator)
                        line = ['<S>'] + line.split() + ['</S>']
                        stream_item.extend(line)
                    _b = stream_item[:unroll_steps]
                    _s = stream_item[unroll_steps:]
                    batch_item.clear()
                    _b = _b
                    batch_item.extend(_b)

                    stream_item.clear()
                    stream_item.extend(_s)
                yield batch
        except StopIteration:
            pass

    def gen_batches(self, batch_size: int, data_type: str = 'train', shuffle: bool = None)\
            -> Iterator[Tuple[str,str]]:
        if shuffle is None:
            shuffle = self.shuffle

        tgt_data = self.data[data_type]
        shard_generator = self._shard_generator(tgt_data, shuffle = False, random = self.np_random)
        line_generator = self._line_generator(shard_generator)

        unroll_steps = self.unroll_steps if data_type == 'train' else 20

        batch_generator = self._batch_generator(line_generator, batch_size * self.n_gpus, unroll_steps)

        for batch in batch_generator:
            batch = [batch, []]
            yield batch
