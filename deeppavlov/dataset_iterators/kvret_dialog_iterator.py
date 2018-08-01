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

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('kvret_dialog_iterator')
class KvretDialogDatasetIterator(DataLearningIterator):
    """
    Inputs data from :class:`~deeppavlov.dataset_readers.dstc2_reader.DSTC2DatasetReader`, constructs dialog history for each turn, generates batches (one sample is a turn).

    Inherits key methods and attributes from :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`.

    Attributes:
        train: list of "train" ``(context, response)`` tuples
        valid: list of "valid" ``(context, response)`` tuples 
        test: list of "test" ``(context, response)`` tuples 
    """
# TODO: write custom batch_generator: order of utterances from one dialogue is presumed
    @staticmethod
    def _dialogs(data):
        dialogs = []
        history = []
        task = None
        for x, y in data:
            if x.get('episode_done'):
                history = []
                dialogs.append((([], [], [], []), ([], [])))
                task = y['task']
            history.append((x, y))
            x['history'] = history[:-1]
            dialogs[-1][0][0].append(x['text'])
            dialogs[-1][0][1].append(x['dialog_id'])
            dialogs[-1][0][2].append(x.get('kb_columns', None))
            dialogs[-1][0][3].append(x.get('kb_items', None))
            dialogs[-1][1][0].append(y['text'])
            dialogs[-1][1][1].append(task)
        return dialogs

    @staticmethod
    def _utterances(data):
        utters = []
        history = []
        for x, y in data:
            if x.get('episode_done'):
                history = []
            history.append((x, y))
            x['history'] = history[:-1]
            x_tuple = (x['text'], x['dialog_id'], x['kb_columns'], x['kb_items'])
            y_tuple = (y['text'], y['task'])
            utters.append((x_tuple, y_tuple))
        return utters

    @overrides
    def split(self, *args, **kwargs):
        self.train = self._utterances(self.train)
        self.valid = self._utterances(self.valid)
        self.test = self._utterances(self.test)
