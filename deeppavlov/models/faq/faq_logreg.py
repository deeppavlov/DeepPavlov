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

import numpy as np
from typing import List
from scipy.sparse import vstack
from scipy.sparse import csr_matrix

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.common.file import save_pickle
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.serializable import Serializable
from sklearn.linear_model import LogisticRegression

logger = get_logger(__name__)


@register("faq_logreg_model")
class FaqLogregModel(Estimator, Serializable):

    def __init__(self, save_path: str = None, load_path: str = None, **kwargs) -> None:
        self.save_path = save_path
        self.load_path = load_path
        if kwargs['mode'] != 'train':
            self.load()

    def __call__(self, q_vect) -> List[str]:

        probs = self.logreg.predict_proba(q_vect)
        answer_ids = np.argmax(probs, axis=1)

        scores = np.round(np.choose(answer_ids, probs.T).tolist(), 2)
        answers = self.logreg.classes_[answer_ids].tolist()

        return answers, scores

    def fit(self, x_train_vects, y_train) -> None:
        if len(x_train_vects) != 0:
            if isinstance(x_train_vects[0], csr_matrix):
                x_train_features = vstack(list(x_train_vects))
            elif isinstance(x_train_vects[0], np.ndarray):
                x_train_features = np.vstack(list(x_train_vects))
            else:
                raise NotImplementedError('Not implemented this type of vectors')
        else:
            raise ValueError("Train vectors can't be empty")

        self.logreg = LogisticRegression()
        self.logreg.fit(x_train_features, list(y_train))


    def save(self) -> None:
        logger.info("Saving faq_logreg_model to {}".format(self.save_path))
        save_pickle(self.logreg, expand_path(self.save_path))


    def load(self) -> None:
        logger.info("Loading faq_logreg_model from {}".format(self.load_path))
        self.logreg = load_pickle(expand_path(self.load_path))
