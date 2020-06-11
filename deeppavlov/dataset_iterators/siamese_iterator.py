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
    """The class contains methods for iterating over a dataset for ranking in training, validation and test mode."""

    def split(self, *args, len_valid=1000, len_test=1000, **kwargs) -> None:
        if len(self.valid) == 0 and len_valid != 0:
            self.random.shuffle(self.train)
            self.valid = self.train[-len_valid:]
            self.train = self.train[:-len_valid]
        if len(self.test) == 0 and len_test != 0:
            self.random.shuffle(self.train)
            self.test = self.train[-len_test:]
            self.train = self.train[:-len_test]
