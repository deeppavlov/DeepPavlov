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

from typing import List


from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
import numpy as np

logger = get_logger(__name__)


@register('sentence2vector_w2v_avg')
class SentenceAvgW2vVectorizer(Component):
    """Sentence vectorizer which produce one vector as average sum of words vectors in sentence"""

    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, questions: List[str], tokens_fasttext_vectors: List) -> List:
        """Vectorize list of sentences

        Parameters:
            questions: list of questions/sentences
            tokens_fasttext_vectors: fasttext vectors for sentences

        Returns:
            List of vectorized sentences
        """

        questions_vectors = []
        for i, q in enumerate(questions):
            q_weights = [1/len(questions[i])]*len(questions[i])
            questions_vectors.append(np.average(tokens_fasttext_vectors[i], weights=q_weights, axis=0))

        return questions_vectors
