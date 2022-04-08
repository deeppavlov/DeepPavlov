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

from logging import getLogger
from operator import itemgetter
from typing import List, Any, Tuple

import numpy as np
import joblib

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


@register('pop_ranker')
class PopRanker(Component):
    """Rank documents according to their tfidf scores and popularities. It is not a standalone ranker,
    it should be used for re-ranking the results of TF-IDF Ranker.

    Based on a Logistic Regression trained on 3 features:

    * tfidf score of the article
    * popularity of the article obtained via Wikimedia REST API as a mean number of views for the period since 2017/11/05 to 2018/11/05
    * multiplication of the two features above

    Args:
        pop_dict_path: a path to json file with article title to article popularity map
        load_path: a path to saved logistic regression classifier
        top_n: a number of doc ids to return
        active: whether to return a number specified by :attr:`top_n` (``True``) or all ids
         (``False``)

    Attributes:
        pop_dict: a map of article titles to their popularity
        mean_pop: mean popularity of all popularities in :attr:`pop_dict`, use it when popularity is not found
        clf: a loaded logistic regression classifier
        top_n: a number of doc ids to return
        active: whether to return a number specified by :attr:`top_n` or all ids

    """

    def __init__(self, pop_dict_path: str, load_path: str, top_n: int = 3, active: bool = True,
                 **kwargs) -> None:
        pop_dict_path = expand_path(pop_dict_path)
        logger.info(f"Reading popularity dictionary from {pop_dict_path}")
        self.pop_dict = read_json(pop_dict_path)
        self.mean_pop = np.mean(list(self.pop_dict.values()))
        load_path = expand_path(load_path)
        logger.info(f"Loading popularity ranker from {load_path}")
        self.clf = joblib.load(load_path)
        self.top_n = top_n
        self.active = active

    def __call__(self, input_doc_ids: List[List[Any]], input_doc_scores: List[List[float]]) -> \
            Tuple[List[List], List[List]]:
        """Get tfidf scores and tfidf ids, re-rank them by applying logistic regression classifier,
        output pop ranker ids and pop ranker scores.

         Args:
            input_doc_ids: top input doc ids of tfidf ranker
            input_doc_scores: top input doc scores of tfidf ranker corresponding to doc ids

        Returns:
            top doc ids of pop ranker and their corresponding scores

        """
        batch_ids = []
        batch_scores = []
        for instance_ids, instance_scores in zip(input_doc_ids, input_doc_scores):
            instance_probas = []
            for idx, score in zip(instance_ids, instance_scores):
                pop = self.pop_dict.get(idx, self.mean_pop)
                features = [score, pop, score * pop]
                prob = self.clf.predict_proba([features])
                instance_probas.append(prob[0][1])

            sort = sorted(enumerate(instance_probas), key=itemgetter(1), reverse=True)
            sorted_probas = [item[1] for item in sort]
            sorted_ids = [instance_ids[item[0]] for item in sort]

            if self.active:
                sorted_ids = sorted_ids[:self.top_n]
                sorted_probas = sorted_probas[:self.top_n]

            batch_ids.append(sorted_ids)
            batch_scores.append(sorted_probas)

        return batch_ids, batch_scores
