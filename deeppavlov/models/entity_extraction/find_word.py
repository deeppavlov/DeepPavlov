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

import itertools
import pickle
from collections import Counter

import numpy as np
import scipy as sp

from deeppavlov.core.commands.utils import expand_path

Sparse = sp.sparse.csr_matrix


class WordSearcher:
    def __init__(self, words_dict_filename: str, ngrams_matrix_filename: str, lang: str = "@en", thresh: int = 1000):
        self.words_dict_filename = words_dict_filename
        self.ngrams_matrix_filename = ngrams_matrix_filename
        if lang == "@en":
            self.letters = "abcdefghijklmnopqrstuvwxyz"
        elif lang == "@ru":
            self.letters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        else:
            raise ValueError(f'Unexpected lang value: "{lang}"')
        self.thresh = thresh
        self.load()
        self.make_ngrams_dicts()

    def load(self):
        with open(str(expand_path(self.words_dict_filename)), "rb") as fl:
            self.words_dict = pickle.load(fl)
        words_list = list(self.words_dict.keys())
        self.words_list = sorted(words_list)

        loader = np.load(str(expand_path(self.ngrams_matrix_filename)), allow_pickle=True)
        self.count_matrix = Sparse((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

    def make_ngrams_dicts(self):
        self.bigrams_dict, self.trigrams_dict = {}, {}
        bigram_combs = list(itertools.product(self.letters, self.letters))
        bigram_combs = ["".join(comb) for comb in bigram_combs]
        trigram_combs = list(itertools.product(self.letters, self.letters, self.letters))
        trigram_combs = ["".join(comb) for comb in trigram_combs]
        for cnt, bigram in enumerate(bigram_combs):
            self.bigrams_dict[bigram] = cnt
        for cnt, trigram in enumerate(trigram_combs):
            self.trigrams_dict[trigram] = cnt + len(bigram_combs)

    def __call__(self, query, tags):
        ngrams_list = []
        for i in range(len(query) - 1):
            ngram = query[i : i + 2].lower()
            if ngram in self.bigrams_dict:
                ngram_id = self.bigrams_dict[ngram]
                ngrams_list.append(ngram_id)
        for i in range(len(query) - 2):
            ngram = query[i : i + 3].lower()
            if ngram in self.trigrams_dict:
                ngram_id = self.trigrams_dict[ngram]
                ngrams_list.append(ngram_id)
        ngrams_with_cnts = Counter(ngrams_list).most_common()
        ngram_ids = [elem[0] for elem in ngrams_with_cnts]
        ngram_cnts = [1 for _ in ngrams_with_cnts]

        indptr = np.array([0, len(ngram_cnts)])
        query_matrix = Sparse(
            (ngram_cnts, ngram_ids, indptr), shape=(1, len(self.bigrams_dict) + len(self.trigrams_dict))
        )

        scores = query_matrix * self.count_matrix
        scores = np.squeeze(scores.toarray())

        if self.thresh >= len(scores):
            o = np.argpartition(-scores, len(scores) - 1)[0:self.thresh]
        else:
            o = np.argpartition(-scores, self.thresh)[0:self.thresh]
        o_sort = o[np.argsort(-scores[o])]
        o_sort = o_sort.tolist()

        found_words = [self.words_list[n] for n in o_sort]
        found_words = [
            word
            for word in found_words
            if (
                word.startswith(query[0])
                and abs(len(word) - len(query)) < 3
                and self.words_dict[word].intersection(tags)
            )
        ]
        return found_words
