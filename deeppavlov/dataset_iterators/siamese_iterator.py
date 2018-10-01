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

from typing import Dict, List, Tuple
import random

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('siamese_iterator')
class SiameseIterator(DataLearningIterator):
    """The class contains methods for iterating over a dataset for ranking in training, validation and test mode.

    Args:
        data: A dictionary containing training, validation and test parts of the dataset obtainable via
            ``train``, ``valid`` and ``test`` keys.
        seed: Random seed.
        shuffle: Whether to shuffle data.
        random_batches: Whether to choose batches randomly or iterate over data sequentally in training mode.
        batches_per_epoch: A number of batches to choose per each epoch in training mode.
            Only required if ``random_batches`` is set to ``True``.
    """

    def __init__(self,
                 data: Dict[str, List],
                 seed: int = None,
                 shuffle: bool = False,
                 random_batches: bool = False,
                 batches_per_epoch: int = None):

        self.random_batches = random_batches
        self.batches_per_epoch = batches_per_epoch

        random.seed(seed)
        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

        super().__init__(self.data, seed=seed, shuffle=shuffle)


    def gen_batches(self, batch_size: int, data_type: str = "train", shuffle: bool = True)->\
            Tuple[List[List[Tuple[int, int]]], List[int]]:
        """Generate batches of inputs and expected outputs to train neural networks.

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching

        Yields:
            A tuple of a batch of inputs and a batch of expected outputs.

            Inputs and expected outputs have different structure and meaning
            depending on class attributes values and ``data_type``.
        """
        data = self.data[data_type]
        if self.random_batches and self.batches_per_epoch is not None and data_type == "train":
            num_steps = self.batches_per_epoch
            assert(batch_size <= len(data))
        else:
            num_steps = len(data) // batch_size
        if data_type == "train":
            if shuffle:
                random.shuffle(data)
            for i in range(num_steps):
                if self.random_batches:
                    context_response_data = random.sample(data, k=batch_size)
                else:
                    context_response_data = data[i * batch_size:(i + 1) * batch_size]
                yield tuple(zip(*context_response_data))
        if data_type in ["valid", "test"]:
            for i in range(num_steps + 1):
                if i < num_steps:
                    context_response_data = data[i * batch_size:(i + 1) * batch_size]
                else:
                    if len(data[i * batch_size:len(data)]) > 0:
                        context_response_data = data[i * batch_size:len(data)]
                yield tuple(zip(*context_response_data))
