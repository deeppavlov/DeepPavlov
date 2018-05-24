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
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('dialog_iterator')
class DialogDatasetIterator(DataLearningIterator):
    """
    Iterates over dialog data,
    generates batches where one sample is one dialog.
    """

    @staticmethod
    def _dialogs(data):
        dialogs = []
        prev_resp_act = None
        for x, y in data:
            if x.get('episode_done'):
                del x['episode_done']
                prev_resp_act = None
                dialogs.append(([], []))
            x['prev_resp_act'] = prev_resp_act
            prev_resp_act = y['act']
            dialogs[-1][0].append(x)
            dialogs[-1][1].append(y)
        return dialogs

    @overrides
    def split(self, *args, **kwargs):
        self.train = self._dialogs(self.train)
        self.valid = self._dialogs(self.valid)
        self.test = self._dialogs(self.test)


@register('dialog_db_result_iterator')
class DialogDBResultDatasetIterator(DataLearningIterator):
    """
    Iterates over dialog data,
    outputs list of db_result fields (if present).
    """
    @staticmethod
    def _db_result(data):
        x, y = data
        if 'db_result' in x:
            return x['db_result']

    @overrides
    def split(self, *args, **kwargs):
        self.train = [(r, "") for r in filter(None, map(self._db_result, self.train))]
        self.valid = [(r, "") for r in filter(None, map(self._db_result, self.valid))]
        self.test = [(r, "") for r in filter(None, map(self._db_result, self.test))]
