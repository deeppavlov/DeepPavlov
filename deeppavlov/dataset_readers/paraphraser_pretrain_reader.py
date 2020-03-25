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

import json
from typing import Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register("paraphraser_pretrain_reader")
class ParaphraserPretrainReader(DatasetReader):
    """The class to read the pretraining dataset for the paraphrase identification task from files."""

    def read(self,
             data_path: str,
             seed: int = None, *args, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the pretraining dataset for the paraphrase identification task from files.

        Args:
            data_path: A path to a folder with dataset files.
            seed: Random seed.
        """

        data_path = expand_path(data_path)
        train_fname = data_path / 'paraphraser_pretrain_train.json'
        test_fname = data_path / 'paraphraser_pretrain_val.json'
        train_data = self.build_data(train_fname)
        test_data = self.build_data(test_fname)
        dataset = {"train": train_data, "valid": test_data, "test": test_data}
        return dataset

    def int_class(self, str_y):
        if str_y == '-1':
            return 0
        else:
            return 1

    def build_data(self, name):
        with open(name) as f:
            data = json.load(f)
        return [([doc['text_1'], doc['text_2']], self.int_class(doc['class'])) for doc in data]
