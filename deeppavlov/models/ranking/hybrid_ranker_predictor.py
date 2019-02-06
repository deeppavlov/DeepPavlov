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
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


@register("hybrid_ranker_predictor")
class HybridRankerPredictor(Component):

    def __init__(self,
                 sample_size,
                 **kwargs):
        self.sample_size = sample_size


    def __call__(self, candidates_batch, preds_batch):
        """
        return list of best responses and its confidences
        """

        responses_batch = []
        responses_preds = []

        for i in range(len(candidates_batch)):
            d = {candidates_batch[i][j]: preds_batch[i][j] for j in range(len(preds_batch[i]))}
            candidates_list = list(set(candidates_batch[i]))
            scores = [d[c] for c in candidates_list]

            # print("len cand, cands:", len(candidates_list), candidates_list)
            # print("len scores, scores:", len(scores), scores)

            sorted_ids = np.flip(np.argsort(scores), -1)
            chosen_index = np.random.choice(sorted_ids[:self.sample_size])  # choose a random answer as the best one

            responses_batch.append(candidates_list[chosen_index])
            responses_preds.append(scores[chosen_index])

        # print("responses", responses_batch)
        # print("responses_scores", responses_preds)

        return responses_batch, responses_preds