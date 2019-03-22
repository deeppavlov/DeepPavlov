# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from logging import getLogger
from typing import List, Tuple, Union

import numpy as np
from scipy.sparse import vstack, csr_matrix
from scipy.sparse.linalg import norm as sparse_norm

from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.common.file import save_pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.serializable import Serializable

logger = getLogger(__name__)


@register("cos_sim_classifier")
class CosineSimilarityClassifier(Estimator, Serializable):
    """
    Classifier based on cosine similarity between vectorized sentences

    Parameters:
        save_path: path to save the model
        load_path: path to load the model
    """

    def __init__(self, top_n: int = 1, save_path: str = None, load_path: str = None, **kwargs) -> None:
        super().__init__(save_path=save_path, load_path=load_path, **kwargs)
        self.top_n = top_n

        self.x_train_features = self.y_train = None

        if kwargs['mode'] != 'train':
            self.load()

    def __call__(self, q_vects: Union[csr_matrix, List]) -> Tuple[List[str], List[int]]:
        """Found most similar answer for input vectorized question

        Parameters:
            q_vects: vectorized questions

        Returns:
            Tuple of Answer and Score
        """

        if isinstance(q_vects[0], csr_matrix):
            q_norm = sparse_norm(q_vects)
            if q_norm == 0.0:
                cos_similarities = np.zeros((q_vects.shape[0], self.x_train_features.shape[0]))
            else:
                norm = q_norm * sparse_norm(self.x_train_features, axis=1)
                cos_similarities = np.array(q_vects.dot(self.x_train_features.T).todense())
                cos_similarities = cos_similarities / norm
        elif isinstance(q_vects[0], np.ndarray):
            q_vects = np.array(q_vects)
            self.x_train_features = np.array(self.x_train_features)
            norm = np.linalg.norm(q_vects) * np.linalg.norm(self.x_train_features, axis=1)
            cos_similarities = q_vects.dot(self.x_train_features.T) / norm
        elif q_vects[0] is None:
            cos_similarities = np.zeros(len(self.x_train_features))
        else:
            raise NotImplementedError('Not implemented this type of vectors')

        # get cosine similarity for each class
        y_labels = np.unique(self.y_train)
        labels_scores = np.zeros((len(cos_similarities), len(y_labels)))
        for i, label in enumerate(y_labels):
            labels_scores[:, i] = np.max([cos_similarities[:, i]
                                          for i, value in enumerate(self.y_train) if value == label], axis=0)

        labels_scores_sum = labels_scores.sum(axis=1, keepdims=True)
        labels_scores = np.divide(labels_scores, labels_scores_sum,
                                  out=np.zeros_like(labels_scores), where=(labels_scores_sum != 0))

        answer_ids = np.argsort(labels_scores)[:, -self.top_n:]

        # generate top_n answers and scores
        answers = []
        scores = []
        for i in range(len(answer_ids)):
            answers.extend([y_labels[id] for id in answer_ids[i, ::-1]])
            scores.extend([np.round(labels_scores[i, id], 2) for id in answer_ids[i, ::-1]])

        return answers, scores

    def fit(self, x_train_vects: Tuple[Union[csr_matrix, List]], y_train: Tuple[str]) -> None:
        """Train classifier

        Parameters:
            x_train_vects: vectorized question for train dataset
            y_train: answers for train dataset

        Returns:
            None
        """
        if isinstance(x_train_vects, tuple):
            if len(x_train_vects) != 0:
                if isinstance(x_train_vects[0], csr_matrix):
                    self.x_train_features = vstack(list(x_train_vects))
                elif isinstance(x_train_vects[0], np.ndarray):
                    self.x_train_features = np.vstack(list(x_train_vects))
                else:
                    raise NotImplementedError('Not implemented this type of vectors')
            else:
                raise ValueError("Train vectors can't be empty")
        else:
            self.x_train_features = x_train_vects

        self.y_train = list(y_train)

    def save(self) -> None:
        """Save classifier parameters"""
        logger.info("Saving faq_model to {}".format(self.save_path))
        save_pickle((self.x_train_features, self.y_train), self.save_path)

    def load(self) -> None:
        """Load classifier parameters"""
        logger.info("Loading faq_model from {}".format(self.load_path))
        self.x_train_features, self.y_train = load_pickle(self.load_path)
