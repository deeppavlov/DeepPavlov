# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
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
from typing import List, Tuple, Union, Dict

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('ubuntu_v2_mt_reader')
class UbuntuV2MTReader(DatasetReader):
    """The class to read the Ubuntu V2 dataset from csv files taking into account multi-turn dialogue ``context``.

    Please, see https://github.com/rkadlec/ubuntu-ranking-dataset-creator.

    Args:
        data_path: A path to a folder with dataset csv files.
        num_context_turns: A maximum number of dialogue ``context`` turns.
        padding: "post" or "pre" context sentences padding
    """

    def read(self, data_path: str,
             num_context_turns: int = 1,
             padding: str = "post",
             *args, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the Ubuntu V2 dataset from csv files taking into account multi-turn dialogue ``context``.

        Args:
            data_path: A path to a folder with dataset csv files.
            num_context_turns: A maximum number of dialogue ``context`` turns.
            padding: "post" or "pre" context sentences padding

        Returns:
            Dictionary with keys "train", "valid", "test" and parts of the dataset as their values
        """

        self.num_turns = num_context_turns
        self.padding = padding
        dataset = {'train': None, 'valid': None, 'test': None}
        train_fname = Path(data_path) / 'train.csv'
        valid_fname = Path(data_path) / 'valid.csv'
        test_fname = Path(data_path) / 'test.csv'
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
                contexts.append(self._expand_context(el[0].split('__eot__'), padding=self.padding))
                responses.append(el[1])
                labels.append(int(el[2]))
        data = [el[0] + [el[1]] for el in zip(contexts, responses)]
        data = list(zip(data, labels))
        return data

    def preprocess_data_validation(self, fname: Union[Path, str]) -> List[Tuple[List[str], int]]:
        contexts = []
        responses = []
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                contexts.append(self._expand_context(el[0].split('__eot__'), padding=self.padding))
                responses.append(el[1:])
        data = [el[0] + el[1] for el in zip(contexts, responses)]
        data = [(el, 1) for el in data]  # NOTE: labels are useless here actually...
        return data

    def _expand_context(self, context: List[str], padding: str) -> List[str]:
        """
        Align context length by using pre/post padding of empty sentences up to ``self.num_turns`` sentences
        or by reducing the number of context sentences to ``self.num_turns`` sentences.

        Args:
            context (List[str]): list of raw context sentences
            padding (str): "post" or "pre" context sentences padding

        Returns:
            List[str]: list of ``self.num_turns`` context sentences
        """
        if padding == "post":
            sent_list = context
            res = sent_list + (self.num_turns - len(sent_list)) * \
                  [''] if len(sent_list) < self.num_turns else sent_list[:self.num_turns]
            return res
        elif padding == "pre":
            # context[-(self.num_turns + 1):-1]  because the last item of `context` is always '' (empty string)
            sent_list = context[-(self.num_turns + 1):-1]
            if len(sent_list) <= self.num_turns:
                tmp = sent_list[:]
                sent_list = [''] * (self.num_turns - len(sent_list))
                sent_list.extend(tmp)
            return sent_list
