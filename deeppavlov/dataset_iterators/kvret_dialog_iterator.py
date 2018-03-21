"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import copy
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_iterator import BasicDatasetIterator


@register('kvret_dialog_iterator')
class KvretDialogDatasetIterator(BasicDatasetIterator):
    @staticmethod
    def _dialogs(data):
        dialogs = []
        history = []
        for x, y in data:
            if x.get('episode_done') is not None:
                history = []
                dialogs.append((([], []), []))
            history.append((copy.deepcopy(x), y))
            x['history'] = history[:-1]
            dialogs[-1][0][0].append(x.pop('text'))
            dialogs[-1][0][1].append(x)
            #dialogs[-1][0].append(x)
            dialogs[-1][1].append(y.pop('text'))
        return dialogs

    @staticmethod
    def _utterances(data):
        #utters = (([], []), [])
        utters = []
        history = []
        for x, y in data:
            if x.get('episode_done'):
                history = []
            history.append((copy.deepcopy(x), y))
            x['history'] = history[:-1]
            #utters[0][0].append(x.pop('text'))
            #utters[0][1].append(x)
            #utters[1].append(y.pop('text'))
            utters.append(((x.pop('text'), x), y.pop('text')))
        return utters

    @overrides
    def split(self, *args, **kwargs):
        self.train = self._utterances(self.train)
        self.valid = self._utterances(self.valid)
        self.test = self._utterances(self.test)
