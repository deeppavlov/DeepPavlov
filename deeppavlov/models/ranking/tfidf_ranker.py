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

from typing import List, Dict, Any, Tuple

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.models.vectorizers.hashing_tfidf_vectorizer import HashingTfIdfVectorizer
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator

logger = get_logger(__name__)


@register("tfidf_ranker")
class TfidfRanker(Estimator):
    """Rank documents according to input strings.

    Args:
        vectorizer: a vectorizer class
        top_n: a number of doc ids to return
        active: whether to return a number specified by :attr:`top_n` (``True``) or all ids
         (``False``)

    Attributes:
        top_n: a number of doc ids to return
        vectorizer: an instance of vectorizer class
        active: whether to return a number specified by :attr:`top_n` or all ids
        tfidf_matrix: a loaded tfidf matrix
        ngram_range: ngram range used when tfidf matrix was created
        hash_size: hash size of the tfidf matrix
        term_freqs: a dictionary with tfidf terms and their frequences
        doc_index: a dictionary of doc ids and corresponding doc titles
        index2doc: inverted :attr:`doc_index`
        iterator: a dataset iterator used for generating batches while fitting the vectorizer

    """

    def get_main_component(self) -> 'TfidfRanker':
        """Temporary stub to run REST API

        Returns:
            self
        """
        return self

    def __init__(self, vectorizer: HashingTfIdfVectorizer, top_n=5, active: bool = True, **kwargs):

        self.top_n = top_n
        self.vectorizer = vectorizer
        self.active = active

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

    def get_index2doc(self) -> Dict[Any, int]:
        """Invert doc_index.

        Returns:
            inverted doc_index dict

        """
        return dict(zip(self.doc_index.values(), self.doc_index.keys()))

    def __call__(self, questions: List[str]) -> Tuple[List[Any], List[float]]:
        """Rank documents and return top n document titles with scores.

        Args:
            questions: list of queries used in ranking

        Returns:
            a tuple of selected doc ids and their scores
        """

        batch_doc_ids, batch_docs_scores = [], []

        q_tfidfs = self.vectorizer(questions)

        for q_tfidf in q_tfidfs:
            scores = q_tfidf * self.tfidf_matrix
            scores = np.squeeze(
                scores.toarray() + 0.0001)  # add a small value to eliminate zero scores

            if self.active:
                thresh = self.top_n
            else:
                thresh = len(self.doc_index)

            if thresh >= len(scores):
                o = np.argpartition(-scores, len(scores) - 1)[0:thresh]
            else:
                o = np.argpartition(-scores, thresh)[0:thresh]
            o_sort = o[np.argsort(-scores[o])]

            doc_scores = scores[o_sort]
            doc_ids = [self.index2doc[i] for i in o_sort]
            batch_doc_ids.append(doc_ids)
            batch_docs_scores.append(doc_scores)

        return batch_doc_ids, batch_docs_scores

    def fit_batches(self, iterator: DataFittingIterator, batch_size: int) -> None:
        """Generate a batch to be fit to a vectorizer.

        Args:
            iterator: an instance of an iterator class
            batch_size: a size of a generated batch

        Returns:
            None

        """
        self.vectorizer.doc_index = iterator.doc2index
        for x, y in iterator.gen_batches(batch_size):
            self.vectorizer.fit_batch(x, y)

    def fit(self) -> None:
        """Pass method to :class:`Chainer`.

        Returns:
            None

        """
        pass

    def save(self) -> None:
        """Pass method to :attr:`vectorizer`.

        Returns:
            None

        """
        self.vectorizer.save()

    def load(self) -> None:
        """Pass method to :attr:`vectorizer`.

        Returns:
            None

        """
        self.vectorizer.load()
