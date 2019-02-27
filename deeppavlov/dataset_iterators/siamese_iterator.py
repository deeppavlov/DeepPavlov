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

from logging import getLogger
from typing import Dict, List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

log = getLogger(__name__)

@register('siamese_iterator')
class SiameseIterator(DataLearningIterator):
    """The class contains methods for iterating over a dataset for ranking in training, validation and test mode.

    Args:
        data: A dictionary containing training, validation and test parts of the dataset obtainable via
            ``train``, ``valid`` and ``test`` keys.
        seed: Random seed.
        shuffle: Whether to shuffle data.
        num_samples: A number of data samples to use in ``train``, ``validation`` and ``test`` mode.
        random_batches: Whether to choose batches randomly or iterate over data sequentally in training mode.
        batches_per_epoch: A number of batches to choose per each epoch in training mode.
            Only required if ``random_batches`` is set to ``True``.
    """

    def __init__(self,
                 data: Dict[str, List],
                 seed: int = None,
                 shuffle: bool = False,
                 num_samples: int = None,
                 random_batches: bool = False,
                 batches_per_epoch: int = None,
                 *args, **kwargs) -> None:

        self.len_valid = kwargs.get("len_valid", 1000)
        self.len_test = kwargs.get("len_test", 1000)
        super().__init__(data, seed=seed, shuffle=shuffle, *args, **kwargs)
        self.random_batches = random_batches
        self.batches_per_epoch = batches_per_epoch
        self.data["train"] = self.train[:num_samples]
        self.data["valid"] = self.valid[:num_samples]
        self.data["test"] = self.test[:num_samples]
        self.data["all"] = self.train + self.valid + self.test

    def split(self, *args, **kwargs) -> None:
        if len(self.valid) == 0:
            self.random.shuffle(self.train)
            self.valid = self.train[-1000:]
            self.train = self.train[:-1000]
        if len(self.test) == 0:
            self.random.shuffle(self.train)
            self.test = self.train[-1000:]
            self.train = self.train[:-1000]

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
            if batch_size > len(data):
                batch_size = len(data)
                log.warning("The batch size exceeds the dataset size. Setting it equal to the dataset size.")
        else:
            num_steps = len(data) // batch_size
        if data_type == "train":
            if shuffle:
                self.random.shuffle(data)
            for i in range(num_steps):
                if self.random_batches:
                    context_response_data = self.random.sample(data, k=batch_size)
                else:
                    context_response_data = data[i * batch_size:(i + 1) * batch_size]
                yield tuple(zip(*context_response_data))
        if data_type in ["valid", "test"]:
            for i in range(num_steps + 1):
                context_response_data = data[i * batch_size:(i + 1) * batch_size]
                if context_response_data != []:
                    yield tuple(zip(*context_response_data))
