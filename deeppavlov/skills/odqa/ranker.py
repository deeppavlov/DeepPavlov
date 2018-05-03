"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Type, List

import numpy as np
from scipy.sparse import csr_matrix

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.models.vectorizers.hashing_tfidf_vectorizer import HashingTfIdfVectorizer

logger = get_logger(__name__)


@register("tfidf_ranker")
class TfidfRanker(Estimator):
    """
    temporary stub to run REST API
     """

    def get_main_component(self):
        return self

    def __init__(self, vectorizer: HashingTfIdfVectorizer, top_n=5, **kwargs):

        self.top_n = top_n
        self.vectorizer = vectorizer

        if kwargs['mode'] != 'train':
            if self.vectorizer.load_path.exists():
                self.tfidf_matrix, opts = self.vectorizer.load()
                self.ngram_range = opts['ngram_range']
                self.hash_size = opts['hash_size']
                self.term_freqs = opts['term_freqs'].squeeze()
                self.doc_index = opts['doc_index']

                self.vectorizer.doc_index = self.doc_index
                self.vectorizer.term_freqs = self.term_freqs
                self.vectorizer.hash_size = self.hash_size

                self.index2doc = self.get_index2doc()
            else:
                self.iterator = None
                logger.warning("TfidfRanker load_path doesn't exist, is waiting for training.")

    def get_index2doc(self):
        return dict(zip(self.doc_index.values(), self.doc_index.keys()))

    def __call__(self, questions: List[str]):
        """
        Rank documents and return top n document titles with scores.
        :param questions: queries to search an answer for
        :param n: a number of documents to return
        :return: document ids, document scores
        """
        batch_doc_ids, batch_docs_scores = [], []

        q_tfidfs = self.vectorizer(questions)

        for q_tfidf in q_tfidfs:
            scores: csr_matrix = q_tfidf * self.tfidf_matrix

            if len(scores.data) <= self.top_n:
                o_sort = np.argsort(-scores.data)
            else:
                o = np.argpartition(-scores.data, self.top_n)[0:self.top_n]
                o_sort = o[np.argsort(-scores.data[o])]

            o_sort = scores.indices[o_sort]
            scores = np.squeeze(scores.toarray())

            # for cases when o_sort is empty
            if len(o_sort) < self.top_n:
                o_sort = np.concatenate([o_sort, np.arange(self.top_n - len(o_sort))])

            doc_scores = scores[o_sort]
            doc_ids = [self.index2doc[i] for i in o_sort]
            batch_doc_ids.append(doc_ids)
            batch_docs_scores.append(doc_scores)

        return batch_doc_ids, batch_docs_scores

    def fit_batches(self, iterator, batch_size: int):
        self.vectorizer.doc_index = iterator.doc2index
        for x, y in iterator.gen_batches(batch_size):
            self.vectorizer.fit_batch(x, y)

    def fit(self):
        pass

    def save(self):
        self.vectorizer.save()

    def load(self):
        self.vectorizer.load()


