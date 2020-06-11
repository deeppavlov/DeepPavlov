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

import warnings
from logging import getLogger
from operator import itemgetter
from typing import List, Union, Tuple

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


@register("logit_ranker")
class LogitRanker(Component):
    """Select best answer using squad model logits. Make several batches for a single batch, send each batch
     to the squad model separately and get a single best answer for each batch.

     Args:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
        sort_noans: whether to downgrade noans tokens in the most possible answers

     Attributes:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model

    """

    def __init__(self, squad_model: Union[Chainer, Component], batch_size: int = 50,
                 sort_noans: bool = False, **kwargs):
        self.squad_model = squad_model
        self.batch_size = batch_size
        self.sort_noans = sort_noans

    def __call__(self, contexts_batch: List[List[str]], questions_batch: List[List[str]]) -> \
            Tuple[List[str], List[float]]:
        """
        Sort obtained results from squad reader by logits and get the answer with a maximum logit.

        Args:
            contexts_batch: a batch of contexts which should be treated as a single batch in the outer JSON config
            questions_batch: a batch of questions which should be treated as a single batch in the outer JSON config

        Returns:
            a batch of best answers and their scores

        """
        # TODO output result for top_n
        warnings.warn(f'{self.__class__.__name__}.__call__() API will be changed in the future release.'
                      ' Instead of returning Tuple(List[str], List[float] will return'
                      ' Tuple(List[List[str]], List[List[float]]).', FutureWarning)

        batch_best_answers = []
        batch_best_answers_scores = []
        for contexts, questions in zip(contexts_batch, questions_batch):
            results = []
            for i in range(0, len(contexts), self.batch_size):
                c_batch = contexts[i: i + self.batch_size]
                q_batch = questions[i: i + self.batch_size]
                batch_predict = zip(*self.squad_model(c_batch, q_batch))
                results += batch_predict
            if self.sort_noans:
                results = sorted(results, key=lambda x: (x[0] != '', x[2]), reverse=True)
            else:
                results = sorted(results, key=itemgetter(2), reverse=True)
            batch_best_answers.append(results[0][0])
            batch_best_answers_scores.append(results[0][2])
        return batch_best_answers, batch_best_answers_scores
