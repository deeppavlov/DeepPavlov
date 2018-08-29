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

from typing import List

from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from sklearn.feature_extraction.text import TfidfVectorizer
from deeppavlov.core.common.file import save_pickle
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.commands.utils import expand_path, make_all_dirs, is_file_exist
import numpy as np

logger = get_logger(__name__)


@register('sentence2vector_w2v_tfidf')
class SentenceW2vVectorizerTfidfWeights(Estimator, Serializable):
    """
    Sentence vectorizer which produce one vector as tf-idf weighted sum of words vectors in sentence

    Parameters:
        save_path: path to save the model
        load_path: path to load the model

    Returns:
        None
    """

    def __init__(self, save_path: str = None, load_path: str = None, **kwargs) -> None:
        self.save_path = save_path
        self.load_path = load_path

        if is_file_exist(self.load_path):
            self.load()
        else:
            if kwargs['mode'] != 'train':
                self.load()
            else:
                self.vectorizer = TfidfVectorizer()

    def __call__(self, questions: List[str], tokens_fasttext_vectors: List) -> List:
        """Vectorize list of sentences

        Parameters:
            questions: list of questions/sentences
            tokens_fasttext_vectors: fasttext vectors for sentences

        Returns:
            List of vectorized sentences
        """
        if isinstance(questions[0], list):
            questions = [' '.join(x) for x in questions]

        q_vects = self.vectorizer.transform(questions)
        questions_vectors = []
        for i, q in enumerate(questions):
            q_weights = []
            for token in q.split():
                if token in self.token2idx:
                    tfidf_vector = q_vects[i, :]
                    q_weights.append(tfidf_vector[0, self.token2idx[token]])
                else:
                    q_weights.append(0)
            if sum(q_weights) == 0:
                questions_vectors.append(None)
            else:
                questions_vectors.append(np.average(tokens_fasttext_vectors[i], weights=q_weights, axis=0))

        return questions_vectors

    def fit(self, x_train: List) -> None:
        """Train tf-idf weights

        Parameters:
            x_train: train sentences

        Returns:
            None
        """
        if isinstance(x_train[0], list):
            x_train = [' '.join(x) for x in x_train]

        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(x_train)
        self.token2idx = self.vectorizer.vocabulary_

    def save(self) -> None:
        """Save model"""
        logger.info("Saving tfidf_vectorizer to {}".format(self.save_path))
        path = expand_path(self.save_path)
        make_all_dirs(path)
        save_pickle(self.vectorizer, path)

    def load(self) -> None:
        """Load model"""
        logger.info("Loading tfidf_vectorizer from {}".format(self.load_path))
        self.vectorizer = load_pickle(expand_path(self.load_path))
        self.token2idx = self.vectorizer.vocabulary_
