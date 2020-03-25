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

import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('ubuntu_dstc7_mt_reader')
class UbuntuDSTC7MTReader(DatasetReader):
    """
    DatasetReader for Ubuntu Dialogue Corpus Dataset (version 3), prepared for DSTC 7 competition Track 1 Subtrack 1.

    https://github.com/IBM/dstc7-noesis

    Args:
        data_path (str): A path to a folder with dataset json files.
        num_context_turns (int): A maximum number of dialogue ``context`` turns.
        num_responses (int): A number of responses for each context; default is equal to all 100 responses,
            it can be reduced to 10 (1 true response + 9 random wrong responses) to adapt with succeeding pipeline
        padding (str): "post" or "pre" context sentences padding
    """

    def read(self,
             data_path: str,
             num_context_turns: int = 10,
             num_responses: int = 100,
             padding: str = "post",
             seed: int = 42,
             *args, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:

        self.num_turns = num_context_turns
        self.padding = padding
        self.num_responses = num_responses
        self.np_random = np.random.RandomState(seed)

        dataset = {}
        dataset["train"] = self._create_dialog_iter(Path(data_path) / 'ubuntu_train_subtask_1.json', "train")
        dataset["valid"] = self._create_dialog_iter(Path(data_path) / 'ubuntu_dev_subtask_1.json', "valid")
        dataset["test"] = self._create_dialog_iter(Path(data_path) / 'ubuntu_test_subtask_1.json', "test")
        return dataset

    def _create_dialog_iter(self, filename, mode="train"):
        """
        Read input json file with test data and transform it to the following format:
        [
            ( [context_utt_1, ..., context_utt_10, response_utt_1, ..., response_utt_N], label ),
            ( [context_utt_1, ..., context_utt_10, response_utt_1, ..., response_utt_N], label ),
            ...
        ]

        where
        * [context_utt_1, ..., context_utt_10, response_utt_1, ..., response_utt_N] - list that consists of
        ``num_context_turn`` utterances, followed by ``num_responses`` responses.
        Where
        * label - label of the sample

        Args:
            filename (Path): filename to read
            mode (str): which dataset to return. Can be "train", "valid" or "test"

        Returns:
             list of contexts and responses with their labels. More details about the format are provided above
        """
        data = []
        with open(filename, encoding='utf-8') as f:
            json_data = json.load(f)
            for entry in json_data:

                dialog = entry
                utterances = []  # all the context sentences
                for msg in dialog['messages-so-far']:
                    utterances.append(msg['utterance'])

                true_response = ""  # true response sentence
                if mode != "test":
                    true_response = dialog['options-for-correct-answers'][0]['utterance']

                fake_responses = []  # rest (wrong) responses
                target_id = ""
                if mode != "test":
                    correct_answer = dialog['options-for-correct-answers'][0]
                    target_id = correct_answer['candidate-id']
                for i, utterance in enumerate(dialog['options-for-next']):
                    if utterance['candidate-id'] != target_id:
                        fake_responses.append(utterance['utterance'])

                # aligned list of context utterances
                expanded_context = self._expand_context(utterances, padding=self.padding)

                if mode == 'train':
                    data.append((expanded_context + [true_response], 1))
                    data.append(
                        (expanded_context + list(self.np_random.choice(fake_responses, size=1)), 0))  # random 1 from 99

                elif mode == 'valid':
                    # NOTE: labels are useless here...
                    data.append((expanded_context + [true_response] + list(
                        self.np_random.choice(fake_responses, self.num_responses - 1)), 0))

                elif mode == 'test':
                    data.append((expanded_context + fake_responses, 0))

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
            sent_list = context[-(self.num_turns + 1):-1]
            if len(sent_list) <= self.num_turns:
                tmp = sent_list[:]
                sent_list = [''] * (self.num_turns - len(sent_list))
                sent_list.extend(tmp)
            return sent_list
