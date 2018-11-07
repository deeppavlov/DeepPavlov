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

@register('file_paths_iterator')
class FilePathsIterator(DataLearningIterator):
    """Dataset iterator for datasetes like 1 Billion Word Benchmark

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
                 *args, **kwargs) -> None:
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
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
            
    def gen_batches(self, batch_size: int, data_type: str = 'train', shuffle: bool = None)\
            -> Iterator[Tuple[str,str]]:
        if shuffle is None:
            shuffle = self.shuffle

        tgt_data = self.data[data_type]
        shard_generator = self._shard_generator(tgt_data, shuffle = False, random = self.np_random)

        for shard in shard_generator:
            if not (batch_size):
                bs = len(shard)
            lines_generator = self._chunk_generator(shard, bs)
            for lines in lines_generator:
                yield (lines, [])