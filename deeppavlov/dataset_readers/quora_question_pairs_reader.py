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
from typing import Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('qqp_reader')
class QuoraQuestionPairsReader(DatasetReader):
    """The class to read the Quora Question Pairs dataset from files.

    Please, see https://www.kaggle.com/c/quora-question-pairs/data.

    Args:
        data_path: A path to a folder with dataset files.
        seed: Random seed.
    """

    def read(self, data_path: str,
             seed: int = None, *args, **kwargs) -> Dict[str, List[Tuple[Tuple[str, str], int]]]:
        data_path = expand_path(data_path)
        fname = data_path / 'train.csv'
        contexts = []
        responses = []
        labels = []
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                contexts.append(el[-3].replace('\n', '').lower())
                responses.append(el[-2].replace('\n', '').lower())
                labels.append(int(el[-1]))
        data = list(zip(contexts, responses))
        data = list(zip(data, labels))
        data = {"train": data,
                "valid": [],
                "test": []}
        return data
