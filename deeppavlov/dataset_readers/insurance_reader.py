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

from pathlib import Path
from typing import Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('insurance_reader')
class InsuranceReader(DatasetReader):
    """The class to read the InsuranceQA V1 dataset from files.

    Please, see https://github.com/shuzi/insuranceQA.
    """

    def read(self, data_path: str, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        """Read the InsuranceQA V1 dataset from files.

        Args:
            data_path: A path to a folder with dataset files.
        """

        data_path = expand_path(data_path)
        dataset = {'train': None, 'valid': None, 'test': None}
        train_fname = data_path / 'insuranceQA-master/V1/question.train.token_idx.label'
        valid_fname = data_path / 'insuranceQA-master/V1/question.dev.label.token_idx.pool'
        test_fname = data_path / 'insuranceQA-master/V1/question.test1.label.token_idx.pool'
        int2tok_fname = data_path / 'insuranceQA-master/V1/vocabulary'
        response2ints_fname = data_path / 'insuranceQA-master/V1/answers.label.token_idx'
        self.int2tok_vocab = self._build_int2tok_vocab(int2tok_fname)
        self.idxs2cont_vocab = self._build_context2toks_vocab(train_fname, valid_fname, test_fname)
        self.response2str_vocab = self._build_response2str_vocab(response2ints_fname)
        dataset["valid"] = self._preprocess_data_valid_test(valid_fname)
        dataset["train"] = self._preprocess_data_train(train_fname)
        dataset["test"] = self._preprocess_data_valid_test(test_fname)

        return dataset

    def _build_context2toks_vocab(self, train_f: Path, val_f: Path, test_f: Path) -> Dict[int, str]:
        contexts = []
        with open(train_f, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            c, _ = eli.split('\t')
            contexts.append(c)
        with open(val_f, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            _, c, _ = eli.split('\t')
            contexts.append(c)
        with open(test_f, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            _, c, _ = eli.split('\t')
            contexts.append(c)
        idxs2cont_vocab = {el[1]: el[0] for el in enumerate(contexts)}
        return idxs2cont_vocab

    def _build_int2tok_vocab(self, fname: Path) -> Dict[int, str]:
        with open(fname, 'r') as f:
            data = f.readlines()
        int2tok_vocab = {int(el.split('\t')[0].split('_')[1]): el.split('\t')[1][:-1] for el in data}
        return int2tok_vocab

    def _build_response2str_vocab(self, fname: Path) -> Dict[int, str]:
        with open(fname, 'r') as f:
            data = f.readlines()
            response2idxs_vocab = {int(el.split('\t')[0]) - 1:
                                       (el.split('\t')[1][:-1]).split(' ') for el in data}
        response2str_vocab = {el[0]: ' '.join([self.int2tok_vocab[int(x.split('_')[1])]
                                               for x in el[1]]) for el in response2idxs_vocab.items()}
        return response2str_vocab

    def _preprocess_data_train(self, fname: Path) -> List[Tuple[List[str], int]]:
        positive_responses_pool = []
        contexts = []
        responses = []
        labels = []
        with open(fname, 'r') as f:
            data = f.readlines()
        for k, eli in enumerate(data):
            eli = eli[:-1]
            q, pa = eli.split('\t')
            q_tok = ' '.join([self.int2tok_vocab[int(el.split('_')[1])] for el in q.split()])
            pa_list = [int(el) - 1 for el in pa.split(' ')]
            pa_list_tok = [self.response2str_vocab[el] for el in pa_list]
            for elj in pa_list_tok:
                contexts.append(q_tok)
                responses.append(elj)
                positive_responses_pool.append(pa_list_tok)
                labels.append(k)
        train_data = list(zip(contexts, responses))
        train_data = list(zip(train_data, labels))
        return train_data

    def _preprocess_data_valid_test(self, fname: Path) -> List[Tuple[List[str], int]]:
        pos_responses_pool = []
        neg_responses_pool = []
        contexts = []
        pos_responses = []
        with open(fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            pa, q, na = eli.split('\t')
            q_tok = ' '.join([self.int2tok_vocab[int(el.split('_')[1])] for el in q.split()])
            pa_list = [int(el) - 1 for el in pa.split(' ')]
            pa_list_tok = [self.response2str_vocab[el] for el in pa_list]
            nas = [int(el) - 1 for el in na.split(' ')]
            nas_tok = [self.response2str_vocab[el] for el in nas]
            for elj in pa_list_tok:
                contexts.append(q_tok)
                pos_responses.append(elj)
                pos_responses_pool.append(pa_list_tok)
                neg_responses_pool.append(nas_tok)
        data = [[el[0]] + el[1] for el in zip(contexts, neg_responses_pool)]
        data = [(el[0], len(el[1])) for el in zip(data, pos_responses_pool)]
        return data
