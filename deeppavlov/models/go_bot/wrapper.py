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

from typing import Iterable

from deeppavlov.core.models.component import Component


class DialogComponentWrapper(Component):

    def __init__(self, component: Component, **kwargs):
        self.component = component

    @staticmethod
    def _get_text(utter):
        return utter['text']

    def __call__(self, batch):
        out = []
        if isinstance(batch[0], Iterable) and not isinstance(batch[0], str):
            for dialog in batch:
                res = self.component([self._get_text(utter) for utter in dialog])
                out.append(res)
        else:
            out = self.component(batch)
        return out

    def fit(self, data):
        self.component.fit([self._get_text(utter)
                            for dialog in data for utter in dialog])

    def save(self, *args, **kwargs):
        self.component.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        self.component.load(*args, **kwargs)
