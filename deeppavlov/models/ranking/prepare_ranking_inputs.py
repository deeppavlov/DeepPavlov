# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

from typing import List, Tuple
from logging import getLogger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


@register("prepare_ranking_inputs")
class PrepareRankingInputs(Component):

    def __init__(self,
                 **kwargs):
        pass


    def __call__(self, context_batch: List[List[str]], skill_responses_batch: List[List[Tuple[str, str]]]) -> \
                    List[List[str]]:
        """
        return batch of model inputs.

        """

        model_inputs = []
        for i in range(len(context_batch)):
            candidates = [v for k, v in skill_responses_batch[i]]
            item = context_batch[i]
            item.extend(candidates)  # append N response candidates to the each context
            model_inputs.append(item)

        # NOTE: model_inputs shape = (batch_size, num_context_turns+num_ranking_samples)
        return model_inputs
