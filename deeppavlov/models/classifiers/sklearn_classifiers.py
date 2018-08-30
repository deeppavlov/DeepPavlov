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
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import vstack
from scipy.sparse import csr_matrix


from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.errors import ConfigError

log = get_logger(__name__)


@register("logistic_regression")
class LogReg(Estimator):
    """
    The class implements the Logistic Regression Classifier from Sklearn library.

    Args:
        save_path (str): save path
        load_path (str): load path
        mode: train/infer trigger
        **kwargs: additional arguments

    Attributes:
        model: Logistic Regression Classifier class from sklearn
    """

    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                 class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                 verbose=0, warm_start=False, n_jobs=1, **kwargs) -> None:
        """
        Initialize Logistic Regression Classifier or load it from load path, if load_path is not None.
        """
        # Classifier parameters
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

        # Parameters for parent classes
        save_path = kwargs.get('save_path', None)
        load_path = kwargs.get('load_path', None)
        mode = kwargs.get('mode', None)

        super().__init__(save_path=save_path, load_path=load_path, mode=mode)

        self.model = None
        self.load()

    def __call__(self, q_vect: List[str]) -> List[float]:
        """
        Infer on the given data. Predicts the class of the sentence.

        Args:
            q_vect: sparse matrix or [n_samples, n_features] matrix

        Returns:
            a list of classes
        """
        answers = self.model.predict(q_vect)
        return answers

    def fit(self, x, y, weights=None, **kwargs):
        """
        Train on the given data (hole dataset).

        Returns:
            None
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
        """
        Save classifier as file with 'pkl' format.
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
        """
        Load classifier from load path. Classifier must be stored as file with 'pkl' format.
        """
        if self.load_path:
            if self.load_path.exists():
                log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
                self.model = joblib.load(self.load_path)
            else:
                log.warning("initializing `{}` from scratch".format('LogisticRegression'))
                self.model = LogisticRegression(self.penalty, self.dual, self.tol, self.C, self.fit_intercept,
                                                self.intercept_scaling, self.class_weight, self.random_state,
                                                self.solver, self.max_iter, self.multi_class, self.verbose,
                                                self.warm_start, self.n_jobs)
        else:
            log.warning("No `load_path` is provided for {}".format(self.__class__.__name__))
            self.model = LogisticRegression(self.penalty, self.dual, self.tol, self.C, self.fit_intercept,
                                            self.intercept_scaling, self.class_weight, self.random_state,
                                            self.solver, self.max_iter, self.multi_class, self.verbose,
                                            self.warm_start, self.n_jobs)


@register("support_vector_classifier")
class Svm(Estimator):
    """
    The class implements the Support Vector Classifier from Sklearn library.

    Args:
        save_path (str): save path
        load_path (str): load path
        mode: train/infer trigger
        **kwargs: additional arguments

    Attributes:
        model: Support Vector Classifier class from sklearn
    """

    def __init__(self, penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr',
                 fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                 max_iter=1000, **kwargs) -> None:
        """
        Initialize Support Vector Classifier or load it from load path, if load_path is not None.
        """
        # Classifier parameters
        self.C = C
        self.tol = tol
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.multi_class = multi_class
        self.intercept_scaling = intercept_scaling
        self.fit_intercept = fit_intercept

        # Parameters for parent classes
        save_path = kwargs.get('save_path', None)
        load_path = kwargs.get('load_path', None)
        mode = kwargs.get('mode', None)

        super().__init__(save_path=save_path, load_path=load_path, mode=mode)

        self.model = None
        self.load()

    def __call__(self, q_vect: List[str]) -> List[float]:
        """
        Infer on the given data. Predicts the class of the sentence.

        Args:
            q_vect: sparse matrix or [n_samples, n_features] matrix

        Returns:
            a list of classes
        """
        classes = self.model.predict(q_vect)
        return classes

    def fit(self, x, y, weights=None, **kwargs):
        """
        Train on the given data (hole dataset).

        Returns:
            None
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
        """
        Save classifier as file with 'pkl' format.
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
        """
        Load classifier from load path. Classifier must be stored as file with 'pkl' format.
        """
        if self.load_path:
            if self.load_path.exists():
                log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
                self.model = joblib.load(self.load_path)
            else:
                log.warning("initializing `{}` from scratch".format('SVC'))
                self.model = LinearSVC(self.penalty, self.loss, self.dual, self.tol, self.C, self.multi_class,
                                       self.fit_intercept, self.intercept_scaling, self.class_weight, self.verbose,
                                       self.random_state, self.max_iter)
        else:
            log.warning("No `load_path` is provided for {}".format(self.__class__.__name__))
            self.model = LinearSVC(self.penalty, self.loss, self.dual, self.tol, self.C, self.multi_class,
                                   self.fit_intercept, self.intercept_scaling, self.class_weight, self.verbose,
                                   self.random_state, self.max_iter)


@register("random_forest")
class RandomForest(Estimator):
    """
    The class implements the Random Forest Classifier from Sklearn library.

    Args:
        save_path (str): save path
        load_path (str): load path
        mode: train/infer trigger
        **kwargs: additional arguments

    Attributes:
        model: Random Forest Classifier class from sklearn
    """

    def __init__(self, n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None, **kwargs) -> None:
        """
        Initialize Random Forest Classifier or load it from load path, if load_path is not None.
        """
        # Classifier parameters
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

        # Parameters for parent classes
        save_path = kwargs.get('save_path', None)
        load_path = kwargs.get('load_path', None)
        mode = kwargs.get('mode', None)

        super().__init__(save_path=save_path, load_path=load_path, mode=mode)

        self.model = None
        self.load()

    def __call__(self, q_vect: List[str]) -> List[float]:
        """
        Infer on the given data. Predicts the class of the sentence.

        Args:
            q_vect: sparse matrix or [n_samples, n_features] matrix

        Returns:
            a list of classes
        """
        classes = self.model.predict(q_vect)
        return classes

    def fit(self, x, y, weights=None, **kwargs):
        """
        Train on the given data (hole dataset).

        Returns:
            None
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
        """
        Save classifier as file with 'pkl' format.
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
        """
        Load classifier from load path. Classifier must be stored as file with 'pkl' format.
        """
        if self.load_path:
            if self.load_path.exists():
                log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
                self.model = joblib.load(self.load_path)
            else:
                log.warning("initializing `{}` from scratch".format('RandomForestClassifier'))
                self.model = RandomForestClassifier(self.n_estimators, self.criterion, self.max_depth,
                                                    self.min_samples_split, self.min_samples_leaf,
                                                    self.min_weight_fraction_leaf, self.max_features,
                                                    self.max_leaf_nodes, self.min_impurity_decrease,
                                                    self.min_impurity_split, self.bootstrap, self.oob_score,
                                                    self.n_jobs, self.random_state, self.verbose, self.warm_start,
                                                    self.class_weight)
        else:
            log.warning("No `load_path` is provided for {}".format(self.__class__.__name__))
            self.model = RandomForestClassifier(self.n_estimators, self.criterion, self.max_depth,
                                                self.min_samples_split, self.min_samples_leaf,
                                                self.min_weight_fraction_leaf, self.max_features,
                                                self.max_leaf_nodes, self.min_impurity_decrease,
                                                self.min_impurity_split, self.bootstrap, self.oob_score,
                                                self.n_jobs, self.random_state, self.verbose, self.warm_start,
                                                self.class_weight)
