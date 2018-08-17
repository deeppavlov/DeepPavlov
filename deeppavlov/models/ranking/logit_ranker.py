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
from operator import itemgetter

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Component

logger = get_logger(__name__)


@register("logit_ranker")
class LogitRanker(Component):
    """Select best answer using squad model logits. Make several batches for a single batch, send each batch
     to the squad model separately and get a single best answer for each batch.

     Args:
        squad_model: a loaded squad model

     Attributes:
        squad_model: a loaded squad model

    """

    def __init__(self, squad_model, **kwargs):
        self.squad_model = squad_model

    def __call__(self, contexts_batch: List[List[str]], questions_batch: List[List[str]]) -> List[str]:
        """
        Sort obtained results from squad reader by logits and get the answer with a maximum logit.

        Args:
            contexts_batch: a batch of contexts which should be treated as a single batch in the outer JSON config
            questions_batch: a batch of questions which should be treated as a single batch in the outer JSON config

        Returns:
            a batch of best answers

        """

        batch_best_answers = []
        for contexts, questions in zip(contexts_batch, questions_batch):
            results = zip(*self.squad_model(contexts, questions))
            best_answer = sorted(results, key=itemgetter(2), reverse=True)[0][0]
            batch_best_answers.append(best_answer)

        return batch_best_answers
