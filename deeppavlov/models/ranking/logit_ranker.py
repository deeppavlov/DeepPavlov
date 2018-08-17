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
from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Component
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.common.file import read_json
from deeppavlov.deep import find_config

logger = get_logger(__name__)


@register("logit_ranker")
class LogitRanker(Component):
    """Select best answer using squad model logits. Make several batches for a single batch, send each batch
     to the squad model separately and get a single best answer for each batch.

     Args:
        squad_model_config: a JSON path to the squad model

     Attributes:
        squad_model: a loaded squad model, ready for inferring

     Raises:
         FileNotFoundError if a relative path to :attr:`squad_model_config` doesn't have "configs" module as root

    """

    def __init__(self, squad_model_config: str, **kwargs):
        if Path(squad_model_config).is_absolute():
            abs_config_path = squad_model_config
        else:
            abs_config_path = Path(__file__, "..", "..", "..", squad_model_config).resolve()
        if not abs_config_path.exists():
            raise FileNotFoundError("A root dir for 'squad_model_config' is 'configs/'")
        self.squad_model = build_model_from_config(read_json(abs_config_path))

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
            results = self.squad_model(zip(contexts, questions))
            best_answer = sorted(results, key=itemgetter(2), reverse=True)[0][0]
            batch_best_answers.append(best_answer)

        return batch_best_answers
