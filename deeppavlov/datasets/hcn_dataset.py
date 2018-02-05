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

from overrides import overrides
from typing import Generator

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset


@register('hcn_dataset')
class HCNDataset(Dataset):

    def __init__(self, data):
        super().__init__(data)

    @overrides
    def split(self, num_train_instances=200, num_test_instances=50):
        self.test = self.train[num_train_instances:num_train_instances+num_test_instances]
        self.train = self.train[:num_train_instances]

    @overrides
    def iter_all(self, data_type: str = 'train') -> Generator:
        data = self.data[data_type]
        for instance in data:
            yield instance


