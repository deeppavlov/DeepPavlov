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
from deeppavlov.core.common.registry import register
from typing import List, Tuple, Any

log = getLogger(__name__)


@register("image_classification_iterator")
class ImageClassificationDatasetIterator:
    """Class gets data dictionary from DatasetReader instance which returns DataLoader instead of dict of data
    Args:
        data: dictionary of data with fields "train", "valid" and "test"
        seed: random seed for iterating
        shuffle: whether to shuffle examples in batches
        *args: arguments
        **kwargs: arguments

    Attributes:
        data: dictionary of data with fields "train", "valid" and "test"
    """

    def __init__(self, data: dict, *args, **kwargs):
        """Initialize dataset using data from DatasetReader"""
        self.data = data

        self.train = self.preprocess(data.get("train", []), *args, **kwargs)
        self.valid = self.preprocess(data.get("valid", []), *args, **kwargs)
        self.test = self.preprocess(data.get("test", []), *args, **kwargs)
        self.split(*args, **kwargs)
        self.data = {
            "train": self.train,
            "valid": self.valid,
            "test": self.test,
        }

    def gen_batches(self, batch_size: int, data_type: str = "train", shuffle: bool = None):
        """Generate batches of inputs and expected output to train neural
        networks.

        Args:
            batch_size: number of samples in batch [ALREADY USED IN DATA READER]
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching [ALREADY USED IN DATA READER]

        Yields:
             a tuple of a batch of inputs and a batch of expected outputs
        """
        data_loader = self.data[data_type]
        n_batches = len(data_loader)
        if n_batches == 0:
            return

        data_iter = iter(data_loader)

        for _ in range(n_batches):
            inputs = data_iter.next()
            yield inputs[0].numpy(), [[inp] for inp in inputs[1].numpy()]
