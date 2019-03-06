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


@register('mailru_reader')
class MailruReader(DatasetReader):
    """The class to read dataset for ranking or paraphrase identification with Siamese networks."""

    def read(self, data_path: str, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the dataset for ranking or paraphrase identification with Siamese networks.

        Args:
            data_path: A path to a folder with dataset files.
        """

        dataset = {'train': None, 'valid': None, 'test': None}
        data_path = expand_path(data_path)
        train_fname = data_path / 'train.tsv'
        valid_fname = data_path / 'dev.tsv'
        test_fname = data_path / 'test.tsv'
        dataset["train"] = self._preprocess_data_train(train_fname)
        dataset["valid"] = self._preprocess_data_valid(valid_fname)
        dataset["test"] = self._preprocess_data_train(test_fname)
        return dataset

    def _preprocess_data_train(self, fname: Path) -> List[Tuple[List[str], int]]:
        data = []
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for el in reader:
                data.append((el[3:], int(el[0])))
        return data

    def _preprocess_data_valid(self, fname: Path) -> List[Tuple[List[str], int]]:
        data = []
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            texts = []
            i = 0
            for el in reader:
                if i == 0:
                    texts = [el[3], el[4]]
                    i = 1
                elif i < 10:
                    texts.append(el[4])
                    i += 1
                else:
                    data.append((texts, 0))
                    texts = [el[3], el[4]]
                    i = 1
            data.append((texts, 0))
        return data


