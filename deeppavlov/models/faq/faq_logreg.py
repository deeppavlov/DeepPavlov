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
from typing import List, Tuple
from scipy.sparse import vstack
from scipy.sparse import csr_matrix

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.file import save_pickle
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.commands.utils import expand_path, make_all_dirs
from deeppavlov.core.models.serializable import Serializable
from sklearn.linear_model import LogisticRegression

logger = get_logger(__name__)


@register("faq_logreg_model")
class FaqLogregModel(Estimator, Serializable):
    """
    FAQ model based on logistic regression
    """
    def __init__(self, c=1, penalty='l2', save_path: str = None, load_path: str = None, **kwargs) -> None:
        """FAQ model based on logistic regression

        Parameters:
            c: regularization strength in logistic regression model
            penalty: regularization penalty type in logistic regression model
            save_path: path where to save model
            load_path: path to model

        Returns:
            None
        """
        self.save_path = save_path
        self.load_path = load_path
        self.c = c
        self.penalty = penalty
        if kwargs['mode'] != 'train':
            self.load()

    def __call__(self, q_vect) -> Tuple[List[str], List[str]]:
        """Found most similar answer for input vectorized question

        Parameters:
            q_vect: vectorized question

        Returns:
            Tuple of Answer and Score
        """

        probs = self.logreg.predict_proba(q_vect)
        answer_ids = np.argmax(probs, axis=1)

        scores = np.round(np.choose(answer_ids, probs.T).tolist(), 2)
        answers = self.logreg.classes_[answer_ids].tolist()

        return answers, scores

    def fit(self, x_train_vects, y_train) -> None:
        """Train FAQ model

        Parameters:
            x_train_vects: vectorized question for train dataset
            y_train: answers for train dataset

        Returns:
            None
        """
        if len(x_train_vects) != 0:
            if isinstance(x_train_vects[0], csr_matrix):
                x_train_features = vstack(list(x_train_vects))
            elif isinstance(x_train_vects[0], np.ndarray):
                x_train_features = np.vstack(list(x_train_vects))
            else:
                raise NotImplementedError('Not implemented this type of vectors')
        else:
            raise ValueError("Train vectors can't be empty")

        self.logreg = LogisticRegression(C=self.c, penalty=self.penalty)
        self.logreg.fit(x_train_features, list(y_train))

    def save(self) -> None:
        """Save FAQ model
        """
        logger.info("Saving faq_logreg_model to {}".format(self.save_path))
        path = expand_path(self.save_path)
        make_all_dirs(path)
        save_pickle(self.logreg, path)

    def load(self) -> None:
        """Load FAQ model
        """
        logger.info("Loading faq_logreg_model from {}".format(self.load_path))
        self.logreg = load_pickle(expand_path(self.load_path))
