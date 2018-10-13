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

from typing import List, Generator

# ##########################################
# module imports:
import numpy as np
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.text import text_to_word_sequence
# ##########################################

from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.commands.utils import is_file_exist


TOKENIZER = None
logger = get_logger(__name__)


@register('word2vec_vectorizer')
class Word2vecVectorizer(Estimator):
    """
    Word2vec vectorizer

    Parameters:
        # TODO: add args descriptions
        save_path: path to save the model
        load_path: path to load the model

    Returns:
        None
    """

    def __init__(self,
                 save_path: str = None,
                 load_path: str = None,
                 retrain: str = False,
                 filters: str = "\t\n,",
                 iter: int = 10,
                 dim: int = 200,
                 sg: int = 1,
                 window: int = 10,
                 min_count: int = 1,
                 workers: int = 8,
                 *args, **kwargs) -> None:

        # Calling superclass constructor we set self.save_path along with self.load_path and create these paths
        super(Word2vecVectorizer, self).__init__(save_path=save_path, load_path=load_path, *args, **kwargs)
        self.load_path = str(self.load_path)
        self.save_path = str(self.save_path)

        self.filters = filters
        self.iter = iter
        self.dim = dim
        self.sg = sg
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.retrain = retrain

        self.model = None

        if is_file_exist(self.load_path):
            self.load()

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        """
        Convert words into vectors

        Parameters:
            inputs: list of words

        Returns:
            list of vectorized words
        """
        if isinstance(inputs[0], list):
            inputs = [' '.join(q) for q in inputs]

        # the model is already loaded
        sentences = []
        for line in inputs:
            sentences.append(text_to_word_sequence(line, filters=self.filters, split=" "))

        vecs = []
        for s in sentences:
            vect_sent = []
            for word in s:
                try:
                    vect_sent.append(self.model[word])
                except KeyError:
                    vect_sent.append(np.random.uniform(-0.25, 0.25, self.dim))
            vecs.append(vect_sent)
        return vecs

    def fit(self, x_train: List[str]) -> None:
        """
        Train word2vec model

        Parameters:
            x_train: list of sentences for train

        Returns:
            None
        """
        if self.model is not None and not self.retrain:
            logger.info("No need to retrain existing Word2vec model")
            return

        logger.info("Collecting data for word2vec training")
        sentences = []
        for sample in x_train:
            for line in sample:
                words_seq = text_to_word_sequence(line, filters=self.filters, split=" ")
                if len(words_seq) > 0:
                    sentences.append(words_seq)
        logger.info('Training word2vec')
        self.model = Word2Vec(sentences,
                         iter=self.iter,
                         size=self.dim,
                         sg=self.sg,
                         window=self.window,
                         min_count=self.min_count,
                         workers=self.workers)

        logger.info("Saving word2vec model to {}".format(self.save_path))
        self.model.save(self.save_path)

    def save(self) -> None:
        """Save Word2vec model"""
        logger.info("Saving word2vec model to {}".format(self.save_path))
        self.model.save(self.save_path)

    def load(self) -> None:
        """Load Word2vec model"""
        logger.info("Loading word2vec model from {}".format(self.load_path))
        self.model = Word2Vec.load(self.load_path)

    def __iter__(self) -> Generator:
        yield from self.model.wv.vocab
