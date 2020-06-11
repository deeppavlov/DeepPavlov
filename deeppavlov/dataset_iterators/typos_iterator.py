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

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('typos_iterator')
class TyposDatasetIterator(DataLearningIterator):
    """Implementation of :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator` used for training
     :class:`~deeppavlov.models.spelling_correction.brillmoore.ErrorModel`

    """

    def split(self, test_ratio: float = 0., *args, **kwargs):
        """Split all data into train and test

        Args:
            test_ratio: ratio of test data to train, from 0. to 1.
        """
        self.train += self.valid + self.test

        split = int(len(self.train) * test_ratio)

        self.random.shuffle(self.train)

        self.test = self.train[:split]
        self.train = self.train[split:]
        self.valid = []
