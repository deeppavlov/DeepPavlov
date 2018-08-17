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
import numpy as np

from typing import List

from pathlib import Path
from sklearn.externals import joblib
from sklearn.linear_model.logistic import LogisticRegression

from scipy.sparse import vstack
from scipy.sparse import csr_matrix
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.errors import ConfigError

log = get_logger(__name__)


@register("logistic_regression")
class LogReg(Estimator):
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

    # def get_main_component(self) -> 'LogReg':
    #     """Temporary stub to run REST API
    #
    #     Returns:
    #         self
    #     """
    #     return self

    def __init__(self, **kwargs) -> None:
        # Parameters for parent classes
        save_path = kwargs.get('save_path', None)
        load_path = kwargs.get('load_path', None)
        mode = kwargs.get('mode', None)

        super().__init__(save_path=save_path, load_path=load_path, mode=mode)

        self.model = None
        self.load()

    def __call__(self, q_vect: List[str]) -> List[float]:
        """Predict class of sentence.

        Args:
            questions: list of queries used in ranking

        Returns:
            a list of classes
        """
        if len(q_vect) != 0:
            if isinstance(q_vect[0], csr_matrix):
                q_vect = vstack(list(q_vect))
            elif isinstance(q_vect[0], np.ndarray):
                q_vect = np.vstack(list(q_vect))
            else:
                raise NotImplementedError('Not implemented this type of vectors')
        else:
            raise ValueError("Train vectors can't be empty")

        probs = self.model.predict_proba(q_vect)
        answer_ids = np.argmax(probs, axis=1)

        answers = self.model.classes_[answer_ids].tolist()

        # my shit
        # classes = self.model.predict(q_vect)
        return answers

    def fit(self, x, y, weights=None, **kwargs):
        """Train the model.

        Returns:
            self

        """
        if len(x) != 0:
            if isinstance(x[0], csr_matrix):
                x_train_features = vstack(list(x))
            elif isinstance(x[0], np.ndarray):
                x_train_features = np.vstack(list(x))
            else:
                raise NotImplementedError('Not implemented this type of vectors')
        else:
            raise ValueError("Train vectors can't be empty")

        self.model.fit(x_train_features, list(y))
        # self.model.fit(x, y, weights, **kwargs)
        return self

    def save(self, fname: str = None) -> None:
        """Pass method to :attr:`vectorizer`.

        Returns:
            None

        """
        if not fname:
            fname = self.save_path
        else:
            fname = Path(fname).resolve()

        if not fname.parent.is_dir():
            raise ConfigError("Provided save path is incorrect!")
        else:
            log.info(f"[saving model to {fname}]")
            joblib.dump(self.model, fname)

    def load(self) -> None:
        """Pass method to :attr:`vectorizer`.

        Returns:
            None

        """
        if self.load_path:
            opt_path = Path("{}.pkl".format(self.load_path))
            if opt_path.exists():
                log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
                self.model = joblib.load(self.opt_path)
            else:
                log.warning("initializing `{}` from scratch".format('LogisticRegression'))
                self.model = LogisticRegression()
        else:
            log.warning("No `load_path` is provided for {}".format(self.__class__.__name__))
            self.model = LogisticRegression()
