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

import re
from collections import defaultdict
from logging import getLogger
from typing import List

import numpy as np
import spacy
from scipy.sparse import csr_matrix

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register

log = getLogger(__name__)


@register('ru_adj_to_noun')
class RuAdjToNoun:
    """
        Class for converting an adjective in Russian to the corresponding noun, for example:
        "московский" -> "Москва", "африканский" -> "Африка"
    """

    def __init__(self, freq_dict_filename: str, candidate_nouns: int = 10, freq_thres: float = 4.5,
                 score_thres: float = 2.8, **kwargs):
        """

        Args:
            freq_dict_filename: file with the dictionary of Russian words with the corresponding frequencies
            candidate_nouns: how many candidate nouns to leave after search
            **kwargs:
        """
        self.candidate_nouns = candidate_nouns
        self.freq_thres = freq_thres
        self.score_thres = score_thres
        alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя-"
        self.alphabet_length = len(alphabet)
        self.max_word_length = 24
        self.letter_nums = {letter: num for num, letter in enumerate(alphabet)}
        with open(str(expand_path(freq_dict_filename)), 'r') as fl:
            lines = fl.readlines()
        pos_freq_dict = defaultdict(list)
        for line in lines:
            line_split = line.strip('\n').split('\t')
            if re.match("[\d]+\.[\d]+", line_split[2]):
                pos_freq_dict[line_split[1]].append((line_split[0], float(line_split[2])))
        self.nouns_with_freq = pos_freq_dict["s.PROP"]
        self.adj_set = set([word for word, freq in pos_freq_dict["a"]])
        self.nouns = [noun[0] for noun in self.nouns_with_freq]
        self.matrix = self.make_sparse_matrix(self.nouns).transpose()
        self.nlp = spacy.load("ru_core_news_sm")

    def search(self, word: str):
        word = self.nlp(word)[0].lemma_
        if word in self.adj_set:
            q_matrix = self.make_sparse_matrix([word])
            scores = q_matrix * self.matrix
            scores = np.squeeze(scores.toarray())
            indices = np.argsort(-scores)[:self.candidate_nouns]
            scores = list(scores[indices])
            candidates = [self.nouns_with_freq[indices[i]] + (scores[i],) for i in range(len(indices))]
            candidates = [cand for cand in candidates if cand[0][:3].lower() == word[:3].lower()]
            candidates = sorted(candidates, key=lambda x: (x[2], x[1]), reverse=True)
            log.debug(f"AdjToNoun, found nouns: {candidates}")
            if candidates and candidates[0][1] > self.freq_thres and candidates[0][2] > self.score_thres:
                return candidates[0][0]
        return ""

    def make_sparse_matrix(self, words: List[str]):
        indptr = []
        indices = []
        data = []

        total_length = 0

        for n, word in enumerate(words):
            indptr.append(total_length)
            for cnt, letter in enumerate(word.lower()):
                col = self.alphabet_length * cnt + self.letter_nums[letter]
                indices.append(col)
                init_value = 1.0 - cnt * 0.05
                if init_value < 0:
                    init_value = 0
                data.append(init_value)
            total_length += len(word)

        indptr.append(total_length)

        data = np.array(data)
        indptr = np.array(indptr)
        indices = np.array(indices)

        matrix = csr_matrix((data, indices, indptr), shape=(len(words), self.max_word_length * self.alphabet_length))

        return matrix
