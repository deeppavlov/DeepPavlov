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

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.file import save_pickle
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.commands.utils import expand_path

TOKENIZER = None
logger = get_logger(__name__)


@register('tfidf_vectorizer')
class TfIdfVectorizer(Estimator, Serializable):
    """
    The class implements the tf-idf vectorizer from Sklearn library.

    Args:
        save_path (str): save path
        load_path (str): load path
        **kwargs: additional arguments

    Attributes:
        vectorizer: tf-idf vectorizer class from sklearn
    """

    def __init__(self, save_path: str = None, load_path: str = None, **kwargs) -> None:
        """
        Initialize tf-idf vectorizer or load it from load path, if load_path is not None.
        """
        self.save_path = save_path
        self.load_path = load_path

        if kwargs['mode'] != 'train':
            self.load()
        else:
            self.vectorizer = TfidfVectorizer()

    def __call__(self, questions: List[str]):
        """
        Infer on the given data

        Args:
            questions: list of text samples

        Returns:
            Tf-idf-weighted document-term matrix:
                sparse matrix, [n_samples, n_features]
        """
        q_vects = self.vectorizer.transform([' '.join(q) for q in questions])
        return q_vects

    def fit(self, x_train: List[str]) -> None:
        """
        Train on the given data (hole dataset).

        Args:
            x_train: list of text samples

        Returns:
            None
        """
        self.vectorizer.fit([' '.join(x) for x in x_train])

    def save(self) -> None:
        """
        Save tf-idf vectorizer as file with 'pkl' format.
        """
        logger.info("Saving tfidf_vectorizer to {}".format(self.save_path))
        save_pickle(self.vectorizer, expand_path(self.save_path))

    def load(self) -> None:
        """
        Load tf-idf vectorizer from load path. Vectorizer must be stored as file with 'pkl' format.
        """
        logger.info("Loading tfidf_vectorizer from {}".format(self.load_path))
        self.vectorizer = load_pickle(expand_path(self.load_path))


@register('count_vectorizer')
class SkcountVectorizer(Estimator, Serializable):
    """
    The class implements the count vectorizer from Sklearn library.

    Args:
        save_path (str): save path
        load_path (str): load path
        **kwargs: additional arguments

    Attributes:
        vectorizer: count vectorizer class from sklearn
    """

    def __init__(self, save_path: str = None, load_path: str = None, **kwargs) -> None:
        """
        Initialize count vectorizer or load it from load path, if load_path is not None.
        """
        self.save_path = save_path
        self.load_path = load_path

        if kwargs['mode'] != 'train':
            self.load()
        else:
            self.vectorizer = CountVectorizer()

    def __call__(self, questions: List[str]):
        """
        Infer on the given data

        Args:
            questions: list of text samples

        Returns:
            Document-term matrix:
                sparse matrix, [n_samples, n_features]
        """
        q_vects = self.vectorizer.transform([' '.join(q) for q in questions])
        return q_vects

    def fit(self, x_train: List[str]) -> None:
        """
        Train on the given data (hole dataset).

        Args:
            x_train: list of text samples

        Returns:
            None
        """
        self.vectorizer.fit([' '.join(x) for x in x_train])

    def save(self) -> None:
        """
        Save count vectorizer as file with 'pkl' format.
        """
        logger.info("Saving tfidf_vectorizer to {}".format(self.save_path))
        save_pickle(self.vectorizer, expand_path(self.save_path))

    def load(self) -> None:
        """
        Load count vectorizer from load path. Vectorizer must be stored as file with 'pkl' format.
        """
        logger.info("Loading tfidf_vectorizer from {}".format(self.load_path))
        self.vectorizer = load_pickle(expand_path(self.load_path))
