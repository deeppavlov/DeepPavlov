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

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.models.vectorizers.tfidf_vectorizer import TfIdfVectorizer
from deeppavlov.core.common.file import save_pickle
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.commands.utils import expand_path

logger = get_logger(__name__)


@register("autofaq")
class AutoFaq(Estimator):

    def __init__(self, vectorizer: TfIdfVectorizer, save_path: str = None, load_path: str = None, **kwargs) -> None:
        self.vectorizer = vectorizer
        self.save_path = save_path
        self.load_path = load_path
        if kwargs['mode'] != 'train':
            self.load()

    def __call__(self, question: List[List[str]]) -> List[str]:
        question = [' '.join(q) for q in question]
        q_vect = self.vectorizer(question)
        answer_id = np.argmax(np.array(q_vect.dot(self.vectorizer.w_matrix.T).todense()))

        return [self.id2answer[answer_id]]

    def fit(self, x_train: List[List[str]], y_train: List[str], *args) -> None:
        self.vectorizer.fit([' '.join(x) for x in x_train])
        self.id2answer = dict(enumerate(y_train))

    def save(self) -> None:
        self.vectorizer.save()
        save_pickle(self.id2answer, expand_path(self.save_path))

    def load(self) -> None:
        self.vectorizer.load()
        self.id2answer = load_pickle(expand_path(self.load_path))


