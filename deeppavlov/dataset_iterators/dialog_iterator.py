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


@register('dialog_indexing_iterator')
class DialogDatasetIndexingIterator(DataLearningIterator):
    """
    Iterates over dialog data,
    generates batches where one sample is one dialog.
    Assigns unique index value to each turn item of each dialog.

    A subclass of :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`.

    Attributes:
        train: list of training dialogs (tuples ``(context, response)``)
        valid: list of validation dialogs (tuples ``(context, response)``)
        test: list of dialogs used for testing (tuples ``(context, response)``)
    """

    Xs_LABEL = 'x'
    Ys_LABEL = 'y'

    @overrides
    def preprocess(self, data, *args, **kwargs):
        dialogs = []
        prev_resp_act = None
        for x, y in data:
            if x.get('episode_done'):
                del x['episode_done']
                prev_resp_act = None
                dialogs.append(([], []))
            x['prev_resp_act'] = prev_resp_act
            prev_resp_act = y['act']

            dialogue_label = str(len(dialogs))
            dialogue_x_item_label = str(len(dialogs[-1][0]))
            dialogue_y_item_label = str(len(dialogs[-1][1]))

            x_item_full_label = f"{self.Xs_LABEL}_{dialogue_label}_{dialogue_x_item_label}"
            y_item_full_label = f"{self.Ys_LABEL}_{dialogue_label}_{dialogue_y_item_label}"

            x['indexed_value'] = x_item_full_label
            y['indexed_value'] = y_item_full_label

            x['dialogue_label'] = dialogue_label
            y['dialogue_label'] = dialogue_label

            dialogs[-1][0].append(x)
            dialogs[-1][1].append(y)
        return dialogs


@register('dialog_iterator')
class DialogDatasetIterator(DataLearningIterator):
    """
    Iterates over dialog data,
    generates batches where one sample is one dialog.

    A subclass of :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`.

    Attributes:
        train: list of training dialogs (tuples ``(context, response)``)
        valid: list of validation dialogs (tuples ``(context, response)``)
        test: list of dialogs used for testing (tuples ``(context, response)``)
    """

    @overrides
    def preprocess(self, data, *args, **kwargs):
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


@register('dialog_db_result_iterator')
class DialogDBResultDatasetIterator(DataLearningIterator):
    """
    Iterates over dialog data,
    outputs list of all ``'db_result'`` fields (if present).

    The class helps to build a list of all ``'db_result'`` values present in a dataset.

    Inherits key methods and attributes from :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`.

    Attributes:
        train: list of tuples ``(db_result dictionary, '')`` from "train" data
        valid: list of tuples ``(db_result dictionary, '')`` from "valid" data
        test: list of tuples ``(db_result dictionary, '')`` from "test" data
    """

    @staticmethod
    def _db_result(data):
        x, y = data
        if 'db_result' in x:
            return x['db_result']

    @overrides
    def preprocess(self, data, *args, **kwargs):
        return [(r, "") for r in filter(None, map(self._db_result, data))]
