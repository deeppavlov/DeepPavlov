"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, softwaredata
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
from itertools import chain
from pathlib import Path

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common import paths

logger = logging.getLogger(__name__)


@register('dstc2_datasetreader')
class DSTC2DatasetReader(DatasetReader):

    _train_fname = 'dstc2-trn.jsonlist'
    _valid_fname = 'dstc2-val.jsonlist'
    _test_fname = 'dstc2-tst.jsonlist'

    @overrides
    def read(self, data_path, dialogs=False):
        def _path(dir_path, fname):
            return Path(dir_path).joinpath(fname).as_posix()

        data = {
            'train': self._read_from_file(_path(data_path, self._train_fname),
                                          dialogs),
            'valid': self._read_from_file(_path(data_path, self._valid_fname),
                                          dialogs),
            'test': self._read_from_file(_path(data_path, self._test_fname),
                                         dialogs)
        }
        self.save_vocab(data, paths.USR_PATH / 'vocab.txt')
        return data

    @classmethod
    def _read_from_file(cls, file_path, dialogs=False):
        """Returns data from single file"""
        logger.info("Reading dialog turns from `{}`.".format(file_path))

        utterances, responses, dialog_indices =\
                cls._get_turns(cls._iter_file(file_path), with_indices=True)

        data = list(map(cls._format_turn, zip(utterances, responses)))

        if dialogs:
            return [data[idx['start']:idx['end']] for idx in dialog_indices]
        return data

    @staticmethod
    def _format_turn(turn):
        x = {'text': turn[0]['text'],
             'intents': turn[0]['dialog_acts']}
        if turn[0].get('db_result') is not None:
            x['db_result'] = turn[0]['db_result']
        if turn[0].get('episode_done'):
            x['episode_done'] = True
        y = {'text': turn[1]['text'],
             'act': turn[1]['dialog_acts'][0]['act']}
        return (x, y)

    @staticmethod
    def _iter_file(file_path):
        for ln in open(file_path, 'rt'):
            if ln.strip():
                yield json.loads(ln)
            else:
                yield {}

    @staticmethod
    def _get_turns(data, with_indices=False):
        def _filter(turn):
            del turn['index']
            return turn

        utterances = []
        responses = []
        dialog_indices = []
        n = 0
        num_dialog_utter, num_dialog_resp = 0, 0
        episode_done = True
        for turn in data:
            if not turn:
                if num_dialog_utter != num_dialog_resp:
                    raise RuntimeError("Datafile in the wrong format.")
                episode_done = True
                n += num_dialog_utter
                dialog_indices.append({
                    'start': n - num_dialog_utter,
                    'end': n,
                })
                num_dialog_utter, num_dialog_resp = 0, 0
            else:
                replica = _filter(turn)
                if 'goals' in replica:
                    if episode_done:
                        replica['episode_done'] = True
                    utterances.append(replica)
                    num_dialog_utter += 1
                else:
                    responses.append(replica)
                    num_dialog_resp += 1
                episode_done = False

        if with_indices:
            return utterances, responses, dialog_indices
        return utterances, responses

    @staticmethod
    @overrides
    def save_vocab(data, fpath):
        with open(fpath, 'w') as f:
            words = sorted(list(set(chain.from_iterable(
                [turn[0]['text'].split()\
                 for dt in ['train', 'test', 'valid'] for turn in data[dt]]))))
            f.write(' '.join(words))
