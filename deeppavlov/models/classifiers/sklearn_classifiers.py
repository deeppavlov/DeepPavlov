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
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.errors import ConfigError

log = get_logger(__name__)


@register("logistic_regression")
class LogReg(Estimator):
    """Logistic Regression classifier.

    Args:


    Attributes:


    """

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
            q_vect: list of queries used in ranking

        Returns:
            a list of classes
        """
        answers = self.model.predict(q_vect)
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
            if self.load_path.exists():
                log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
                self.model = joblib.load(self.load_path)
            else:
                log.warning("initializing `{}` from scratch".format('LogisticRegression'))
                self.model = LogisticRegression()
        else:
            log.warning("No `load_path` is provided for {}".format(self.__class__.__name__))
            self.model = LogisticRegression()


@register("support_vector_classifier")
class Svm(Estimator):
    """Rank documents according to input strings.

    Args:


    Attributes:


    """

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
            q_vect: list of queries used in ranking

        Returns:
            a list of classes
        """
        classes = self.model.predict(q_vect)
        return classes

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
            if self.load_path.exists():
                log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
                self.model = joblib.load(self.load_path)
            else:
                log.warning("initializing `{}` from scratch".format('SVC'))
                self.model = LinearSVC()
        else:
            log.warning("No `load_path` is provided for {}".format(self.__class__.__name__))
            self.model = LinearSVC()


@register("random_forest")
class RandomForest(Estimator):
    """Rank documents according to input strings.

    Args:


    Attributes:


    """

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
            q_vect: list of queries used in ranking

        Returns:
            a list of classes
        """
        classes = self.model.predict(q_vect)
        return classes

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
            if self.load_path.exists():
                log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
                self.model = joblib.load(self.load_path)
            else:
                log.warning("initializing `{}` from scratch".format('RandomForestClassifier'))
                self.model = RandomForestClassifier()
        else:
            log.warning("No `load_path` is provided for {}".format(self.__class__.__name__))
            self.model = RandomForestClassifier()
