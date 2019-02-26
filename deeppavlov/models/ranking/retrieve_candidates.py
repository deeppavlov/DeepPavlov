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
import pickle

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


@register("retrieve_candidates")
class RetrieveCandidates(Component):

    def __init__(self,
                 map_filename: str = None,
                 **kwargs):
        map_filename = expand_path(map_filename)
        self.map = pickle.load(open(map_filename, 'rb'))


    def __call__(self, context_batch, index_batch, scores_batch):
        """
        Get batch of contexts, retrieve their corresponded responses,
        return batch of lists of candidates and batch of model inputs.

        context: List[List[str]]
        index: List[List[str]]
        """

        candidates_batch = []  # batch of list of candidates
        for idx, index in enumerate(index_batch):
            ids = [int(i.split('.')[0]) for i in index]
            candidates = [self.map[id][1] for id in ids]

            candidates2 = [(kk, self.map[id][1]) for kk,id in enumerate(ids)]
            contexts = [(kk, self.map[id][0]) for kk,id in enumerate(ids)]
            scores = [(kk,"{:.2f}".format(j)) for kk,j in enumerate(scores_batch[idx])]
            # print("[tf-idf] \nscores:", scores, "\ncontext:", contexts, "\nresponses:", candidates2)  # DEBUG

            candidates_batch.append(candidates)

        model_inputs = []
        for i in range(len(context_batch)):
            item = context_batch[i]
            item.extend(candidates_batch[i])  # append several response candidates to the each context
            model_inputs.append(item)

        # NOTE: candidates_batch shape = (batch_size, num_ranking_samples)
        # NOTE: model_inputs shape = (batch_size, num_context_turns+num_ranking_samples)

        return candidates_batch, model_inputs
