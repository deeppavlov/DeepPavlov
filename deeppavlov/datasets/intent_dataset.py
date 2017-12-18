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


import numpy as np
from sklearn.model_selection import train_test_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset

@register('intent_dataset')
class IntentDataset(Dataset):

    def extract_classes(self, *args, **kwargs):
        intents = []
        all_data = self.iter_all(data_type='train')
        for sample in all_data:
            intents.extend(sample[1])
        if 'valid' in self.data.keys():
            all_data = self.iter_all(data_type='valid')
            for sample in all_data:
                intents.extend(sample[1])
        intents = np.unique(intents)
        return np.array(sorted(intents))

    def split_data(self, field_to_split, new_fields, proportions):
        data_to_div = self.data[field_to_split].copy()
        data_size = len(self.data[field_to_split])
        for i in range(len(new_fields) - 1):
            self.data[new_fields[i]], data_to_div = train_test_split(data_to_div,
                                                                     test_size=len(data_to_div) -
                                                                               int(data_size * proportions[i]))
        self.data[new_fields[-1]] = data_to_div
        return True

    def merge_data(self, fields_to_merge, new_field):
        data = self.data.copy()
        data[new_field] = []
        for name in fields_to_merge:
            data[new_field] += self.data[name]
        self.data = data
        return True
