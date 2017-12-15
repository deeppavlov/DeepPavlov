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

import json
import logging
import itertools
from overrides import overrides
import os
from deeppavlov.data.utils import is_done, mark_done, download_untar

from deeppavlov.core.common.registry import register_model
from deeppavlov.core.data.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)


@register_model('dstc2_reader')
class DSTC2DatasetReader(DatasetReader):

    @staticmethod
    def build(data_path: str):
        data_path = os.path.join(data_path, 'dstc2')
        if not is_done(data_path):

            url = 'http://lnsigo.mipt.ru/export/datasets/dstc2.tar.gz'
            print('Loading DSTC2 from: {}'.format(url))
            download_untar(url, data_path)
            mark_done(data_path)
            print('DSTC2 dataset is built in {}'.format(data_path))
        return os.path.join(data_path, 'dstc2-trn.jsonlist')

    @overrides
    def read(self, data_path='data/'):
        file_path = DSTC2DatasetReader.build(data_path)
        logger.info("Reading instances from lines in file at: {}".format(file_path))
        utterances, responses, dialog_indices =\
                self._read_turns(file_path, with_indices=True)

        data = [{'context': {'text': u['text'],
                             'intents': u['dialog_acts'],
                             'db_result': u.get('db_result', None)},
                 'response': {'text': r['text'],
                              'act': r['dialog_acts'][0]['act']}}\
                for u, r in zip(utterances, responses)]
        return [data[idx['start']:idx['end']] for idx in dialog_indices]

    @staticmethod
    def _read_turns(file_path, with_indices=False):
        def _filter(turn):
            del turn['index']
            return turn

        utterances = []
        responses = []
        dialog_indices = []
        n = 0
        num_dial_utter, num_dial_resp = 0, 0
        for ln in open(file_path, 'rt'):
            if not ln.strip():
                if num_dial_utter != num_dial_resp:
                    raise RuntimeError("Datafile in the wrong format.")
                n += num_dial_utter
                dialog_indices.append({
                    'start': n - num_dial_utter,
                    'end': n,
                })
                num_dial_utter, num_dial_resp = 0, 0
            else:
                replica = _filter(json.loads(ln))
                if 'goals' in replica:
                    utterances.append(replica)
                    num_dial_utter += 1
                else:
                    responses.append(replica)
                    num_dial_resp += 1

        if with_indices:
            return utterances, responses, dialog_indices
        return utterances, responses

    @staticmethod
    @overrides
    def save_vocab(dialogs, fpath):
        with open(fpath, 'wt') as f:
            words = sorted(set(itertools.chain.from_iterable(
                turn['context']['text'].lower().split()\
                for dialog in dialogs for turn in dialog
            )))
            f.write(' '.join(words))