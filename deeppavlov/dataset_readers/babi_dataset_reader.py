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

import logging
from itertools import chain
from pathlib import Path

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.common import paths
from deeppavlov.core.data.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)


@register('babi')
class BabiDatasetReader(DatasetReader):
    def __init__(self):
        pass

    @overrides
    def read(self, file_path):

        logger.info("Reading instances from lines in file at: {}".format(file_path))
        dialogs, dialog_indices = self._read_dialogs(file_path, with_indices=True)

        # get utterances
        utterances = self._get_utterances(file_path, dialogs)
        # get responses
        responses = self._get_responses(file_path, dialogs)

        responses_path = Path(paths.deeppavlov_root) / 'responses.txt'
        responses_path.write_text('\n'.join(responses))

        trainset = [{'context': u, 'response': r} for u, r in zip(utterances, responses)]

        res = []

        for i, dialog_idx in enumerate(dialog_indices):
            # get start and end index
            start, end = dialog_idx['start'], dialog_idx['end']
            # train on dialogue
            whole_dialog = trainset[start:end]
            res.append(whole_dialog)

        self.save_vocab(res, paths.deeppavlov_root / 'vocab.txt')
        return {'train': res}

    @staticmethod
    def _read_dialogs(file_path, with_indices=False):
        def rm_index(row):
            return [' '.join(row[0].split(' ')[1:])] + row[1:]

        def filter_(dialogs):
            filtered_ = []
            for row in dialogs:
                if row[0][:6] != 'resto_':
                    filtered_.append(row)
            return filtered_

        with open(file_path) as f:
            dialogs = filter_([rm_index(row.split('\t')) for row in f.read().split('\n')])
            # organize dialogs -> dialog_indices
            prev_idx = -1
            n = 1
            dialog_indices = []
            updated_dialogs = []
            for i, dialog in enumerate(dialogs):
                if not dialogs[i][0]:
                    dialog_indices.append({
                        'start': prev_idx + 1,
                        'end': i - n + 1
                    })
                    prev_idx = i - n
                    n += 1
                else:
                    updated_dialogs.append(dialog)

            if with_indices:
                return updated_dialogs, dialog_indices[:-1]

            return updated_dialogs

    def _get_utterances(self, file_path, dialogs=None):
        if dialogs is None:
            dialogs = []
        dialogs = dialogs if len(dialogs) else self._read_dialogs(file_path)
        return [row[0] for row in dialogs]

    def _get_responses(self, file_path, dialogs=None):
        if dialogs is None:
            dialogs = []
        dialogs = dialogs if len(dialogs) else self._read_dialogs(file_path)
        return [row[1] for row in dialogs]

#TODO: move save_vocab to babi_dataset
    @staticmethod
    def save_vocab(dialogs, fpath):
        with open(fpath, 'w') as f:
            words = sorted(list(set(chain.from_iterable(
                [instance['context'].split() for dialog in dialogs for instance in dialog]))))
            f.write(' '.join(words))
