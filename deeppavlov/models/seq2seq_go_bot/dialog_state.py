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

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register("dialog_state")
class DialogState(Component):
    def __init__(self, *args, **kwargs):
        self.states = {}

    def __call__(self, user_ids, utterances=None, *args, **kwargs):
        if utterances is None:
            return [self.states.get(u, []) for u in user_ids]

        for user, utter in zip(user_ids, utterances):
            self.states[user] = self.states.get(user, []) + utter
        return
