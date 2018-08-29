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
from deeppavlov.core.common.file import save_pickle
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.commands.utils import expand_path, make_all_dirs, is_file_exist
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


TOKENIZER = None
logger = get_logger(__name__)


@register('tfidf_vectorizer')
class TfIdfVectorizer(Estimator, Serializable):
    """
    Sentence vectorizer which produce sparse vector with TF-IDF values for each word in sentence

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
            if kwargs['mode'] == 'train':
                self.vectorizer = TfidfVectorizer()
            else:
                self.load()

    def __call__(self, questions: List[str]) -> csr_matrix:
        """
        Vectorize sentence into TF-IDF values

        Parameters:
            questions: list of sentences

        Returns:
            list of vectorized sentences
        """
        if isinstance(questions[0], list):
            questions = [' '.join(q) for q in questions]

        q_vects = self.vectorizer.transform(questions)
        return q_vects

    def fit(self, x_train: List[str]) -> None:
        """
        Train TF-IDF vectorizer

        Parameters:
            x_train: list of sentences for train

        Returns:
            None
        """
        if isinstance(x_train[0], list):
            x_train = [' '.join(q) for q in x_train]

        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(x_train)

    def save(self) -> None:
        """Save TF-IDF vectorizer"""
        path = expand_path(self.save_path)
        make_all_dirs(path)
        logger.info("Saving tfidf_vectorizer to {}".format(path))
        save_pickle(self.vectorizer, path)

    def load(self) -> None:
        """Load TF-IDF vectorizer"""
        logger.info("Loading tfidf_vectorizer from {}".format(expand_path(self.load_path)))
        self.vectorizer = load_pickle(expand_path(self.load_path))
