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

from typing import List, Tuple, Union

import numpy as np
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.file import save_pickle
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.commands.utils import expand_path, make_all_dirs
from deeppavlov.core.models.serializable import Serializable

logger = get_logger(__name__)


@register("logreg_classifier")
class LogregClassifier(Estimator, Serializable):
    """
    Logistic Regression Classifier

    Parameters:
        top_n: how many top answers classifier'll return for input vectorized question
        c: regularization strength in logistic regression model
        penalty: regularization penalty type in logistic regression model
        save_path: path to save the model
        load_path: path to load the model

    Returns:
        None
    """
    def __init__(self, top_n: int = 1, c: int = 1, penalty: str = 'l2', save_path: str = None, load_path: str = None, **kwargs) -> None:
        self.save_path = save_path
        self.load_path = load_path
        self.top_n = top_n
        self.c = c
        self.penalty = penalty
        if kwargs['mode'] != 'train':
            self.load()

    def __call__(self, q_vects: List) -> Tuple[List[str], List[int]]:
        """Found most similar answer for input vectorized questions

        Parameters:
            q_vects: vectorized questions

        Returns:
            Tuple of Answer and Score
        """

        probs = self.logreg.predict_proba(q_vects)
        answer_ids = np.argsort(probs)[:, -self.top_n:]

        answers = []
        scores = []
        for i in range(len(answer_ids)):
            answers.extend([self.logreg.classes_[id] for id in answer_ids[i, ::-1]])
            scores.extend([np.round(probs[i, id], 2) for id in answer_ids[i, ::-1]])

        return answers, scores

    def fit(self, x_train_vects: Tuple[Union[csr_matrix, List]], y_train: Tuple[str]) -> None:
        """Train classifier

        Parameters:
            x_train_vects: vectorized questions for train dataset
            y_train: answers for train dataset

        Returns:
            None
        """
        if isinstance(x_train_vects, tuple):
            if len(x_train_vects) != 0:
                if isinstance(x_train_vects[0], csr_matrix):
                    x_train_features = vstack(list(x_train_vects))
                elif isinstance(x_train_vects[0], np.ndarray):
                    x_train_features = np.vstack(list(x_train_vects))
                else:
                    raise NotImplementedError('Not implemented this type of vectors')
            else:
                raise ValueError("Train vectors can't be empty")
        else:
            x_train_features = x_train_vects

        self.logreg = LogisticRegression(C=self.c, penalty=self.penalty)
        self.logreg.fit(x_train_features, list(y_train))

    def save(self) -> None:
        """Save classifier parameters"""
        logger.info("Saving faq_logreg_model to {}".format(self.save_path))
        path = expand_path(self.save_path)
        make_all_dirs(path)
        save_pickle(self.logreg, path)

    def load(self) -> None:
        """Load classifier parameters"""
        logger.info("Loading faq_logreg_model from {}".format(self.load_path))
        self.logreg = load_pickle(expand_path(self.load_path))
