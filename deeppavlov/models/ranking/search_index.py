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

import pickle

from copy import copy

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.estimator import Component

logger = get_logger(__name__)


@register("search_index")
class SearchIndex(Component):

    def __init__(self,
                 map_filename: str = None,
                 **kwargs):
        map_filename = expand_path(map_filename)
        self.map = pickle.load(open(map_filename, 'rb'))


    def __call__(self, context, index):
        ids = [int(i.split('.')[0]) for i in index[0]]
        candidates = [self.map[id][1] for id in ids]


        # TODO: remove: now we use only 1 last replica
        context_ = copy(context)
        context_[0] = context_[0].replace('&', '')
        context_[0] = "&&&&&&&&&" + context_[0]

        # concat context and response candidates
        smn_resp_inputs = context_[0]
        for r in candidates:
            smn_resp_inputs += " & " + r.replace('&', '')

        return [candidates], [smn_resp_inputs]
