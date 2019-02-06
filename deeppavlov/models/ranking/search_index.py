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


@register("search_index")
class SearchIndex(Component):

    def __init__(self,
                 map_filename: str = None,
                 last_turn_only: bool = True,
                 **kwargs):
        map_filename = expand_path(map_filename)
        self.map = pickle.load(open(map_filename, 'rb'))
        self.last_turn_only = last_turn_only


    def __call__(self, context, index):
        ids = [int(i.split('.')[0]) for i in index[0]]
        candidates = [self.map[id][1] for id in ids]

        context_ = list(context)

        if self.last_turn_only:
            # TODO: remove: now we use only 1 last replica
            # context_ does contain a last utterance only like "Hello, bot!"
            context_[0] = context_[0].replace('&', '')  # clean utterance
            context_[0] = "&&&&&&&&&" + context_[0]
            # ############################################
        else:
            # context_ does contain last utterance and context, like " & & & & & & & & Hello, bot! & Hello, human!"
            pass

        # print("[search_index] full_context:", context_[0])

        # append several response candidates to the context
        smn_resp_inputs = context_[0]
        for r in candidates:
            smn_resp_inputs += " & " + r.replace('&', '')

        return [candidates], [smn_resp_inputs]
