"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, softwaredata
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path
import copy

import numpy as np
from sklearn.model_selection import train_test_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset
from deeppavlov.core.common import paths
from deeppavlov.models.preprocessors.preprocessors import PREPROCESSORS


@register('intent_dataset')
class IntentDataset(Dataset):
    def __init__(self, data, seed=None, *args, **kwargs):

        super().__init__(data, seed)

        new_data = {'train': [],
                    'valid': [],
                    "test": []}
        columns = list(data["train"].columns)

        for field in ['train', 'valid', 'test']:
            for sample in self.data[field]:
                new_data[field].append((sample['text'], list(columns[data.loc[0, columns] == 1.0])))
        self.data = new_data





