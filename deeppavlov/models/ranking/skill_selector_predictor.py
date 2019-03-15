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
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


@register("skill_selector_predictor")
class SkillSelectorPredictor(Component):

    def __init__(self,
                 **kwargs):
        pass


    def __call__(self, skill_responses_batch: List[List[Tuple[str, str]]], preds_batch: List[List[float]]) -> \
            List[str]:
        """
        return list of best responses and its confidences
        """
        chosen_skills_batch = []

        # Watch only real responses (not padded ones)
        for i in range(len(skill_responses_batch)):
            scores = preds_batch[i]
            sorted_ids = np.flip(np.argsort(scores), -1)

            # debug
            sorted_scores = [scores[i] for i in sorted_ids]            # [0.9, 0.6, 0.55, 0.54, 0.4, 0.33, 0.32, 0.3]
            # filtered_score_ids = np.where(np.array(sorted_scores) >= 0.5)  # array([0, 1, 2, 3]),)


            chosen_index = sorted_ids[0]
            skill_name = skill_responses_batch[i][chosen_index][0]
            answer = skill_responses_batch[i][chosen_index][-1]

            # logger.debug('candidates: ' + str([candidates_list[i] for i in sorted_ids[:17]]) + 'scores: ' +
            #              str([sorted_scores[i] for i in range(17)]))  # DEBUG
            # logger.debug('answer: ' + str(chosen_index) + " ; " + str(scores[chosen_index]) + " ; " +
            #              str(candidates_list[chosen_index]))  # DEBUG

            chosen_skills_batch.append(skill_name)

        return chosen_skills_batch