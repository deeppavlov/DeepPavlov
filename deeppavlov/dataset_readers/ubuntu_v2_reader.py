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
from typing import List, Dict, Tuple, Union

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('ubuntu_v2_reader')
class UbuntuV2Reader(DatasetReader):
    """The class to read the Ubuntu V2 dataset from csv files.

    Please, see https://github.com/rkadlec/ubuntu-ranking-dataset-creator.
    """

    def read(self, data_path: str,
             positive_samples=False,
             *args, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the Ubuntu V2 dataset from csv files.

        Args:
            data_path: A path to a folder with dataset csv files.
            positive_samples: if `True`, only positive context-response pairs will be taken for train
        """

        data_path = expand_path(data_path)
        dataset = {'train': None, 'valid': None, 'test': None}
        train_fname = Path(data_path) / 'train.csv'
        valid_fname = Path(data_path) / 'valid.csv'
        test_fname = Path(data_path) / 'test.csv'
        self.positive_samples = positive_samples
        self.sen2int_vocab = {}
        self.classes_vocab_train = {}
        self.classes_vocab_valid = {}
        self.classes_vocab_test = {}
        dataset["train"] = self.preprocess_data_train(train_fname)
        dataset["valid"] = self.preprocess_data_validation(valid_fname)
        dataset["test"] = self.preprocess_data_validation(test_fname)
        return dataset

    def preprocess_data_train(self, train_fname: Union[Path, str]) -> List[Tuple[List[str], int]]:
        contexts = []
        responses = []
        labels = []
        with open(train_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                contexts.append(el[0])
                responses.append(el[1])
                labels.append(int(el[2]))
            data = list(zip(contexts, responses))
            data = list(zip(data, labels))
            if self.positive_samples:
                data = [el[0] for el in data if el[1] == 1]
                data = list(zip(data, range(len(data))))
        return data

    def preprocess_data_validation(self, fname: Union[Path, str]) -> List[Tuple[List[str], int]]:
        contexts = []
        responses = []
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                contexts.append(el[0])
                responses.append(el[1:])
        data = [[el[0]] + el[1] for el in zip(contexts, responses)]
        data = [(el, 1) for el in data]
        return data
