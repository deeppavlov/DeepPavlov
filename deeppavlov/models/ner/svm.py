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

import pickle
from itertools import chain
from logging import getLogger
from typing import List, Union

import numpy as np
from sklearn.svm import SVC

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator

log = getLogger(__name__)


@register('ner_svm')
class SVMTagger(Estimator):
    """
    ``SVM`` (Support Vector Machines) classifier for tagging sequences

    Parameters:
        return_probabilities: whether to return probabilities or predictions
        kernel: kernel of SVM (RBF works well in the most of the cases)
        seed: seed for SVM initialization
    """

    def __init__(self, return_probabilities: bool = False, kernel: str = 'rbf', seed=42, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classifier = None
        self.return_probabilities = return_probabilities
        self._kernel = kernel
        self._seed = seed

        self.load()

    def fit(self, tokens: List[List[str]], tags: List[List[int]], *args, **kwargs) -> None:
        tokens = list(chain(*tokens))
        tags = list(chain(*tags))
        self.classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                              decision_function_shape='ovr', degree=3, gamma='auto',
                              kernel=self._kernel, max_iter=-1, probability=self.return_probabilities,
                              random_state=self._seed, shrinking=True, tol=0.001, verbose=False)
        self.classifier.fit(tokens, tags)

    def __call__(self, token_vectors_batch: List[List[str]], *args, **kwargs) -> \
            Union[List[List[int]], List[List[np.ndarray]]]:
        lens = [len(utt) for utt in token_vectors_batch]
        token_vectors_list = list(chain(*token_vectors_batch))
        predictions = self.classifier.predict(token_vectors_list)
        y = []
        cl = 0
        for l in lens:
            y.append(predictions[cl: cl + l])
            cl += l
        return y

    def save(self) -> None:
        with self.save_path.open('wb') as f:
            pickle.dump(self.classifier, f, protocol=4)

    def serialize(self):
        return pickle.dumps(self.classifier, protocol=4)

    def load(self) -> None:
        if self.load_path.exists():
            with self.load_path.open('rb') as f:
                self.classifier = pickle.load(f)

    def deserialize(self, data):
        self.classifier = pickle.loads(data)
