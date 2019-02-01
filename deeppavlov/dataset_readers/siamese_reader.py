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

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('siamese_reader')
class SiameseReader(DatasetReader):
    """The class to read dataset for ranking or paraphrase identification with Siamese networks."""

    def read(self, data_path: str, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the dataset for ranking or paraphrase identification with Siamese networks.

        Args:
            data_path: A path to a folder with dataset files.
        """

        dataset = {'train': None, 'valid': None, 'test': None}
        data_path = expand_path(data_path)
        train_fname = data_path / 'train.csv'
        valid_fname = data_path / 'valid.csv'
        test_fname = data_path / 'test.csv'
        dataset["train"] = self._preprocess_data_train(train_fname)
        dataset["valid"] = self._preprocess_data_valid_test(valid_fname)
        dataset["test"] = self._preprocess_data_valid_test(test_fname)
        return dataset

    def _preprocess_data_train(self, fname: Path) -> List[Tuple[List[str], int]]:
        data = []
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                data.append((el[:2], int(el[2])))
        return data

    def _preprocess_data_valid_test(self, fname: Path) -> List[Tuple[List[str], int]]:
        data = []
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                data.append((el, 1))
        return data
