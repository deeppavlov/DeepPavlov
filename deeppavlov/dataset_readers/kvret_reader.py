# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from logging import getLogger
from pathlib import Path
from typing import Dict, List

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress, mark_done

log = getLogger(__name__)


@register('kvret_reader')
class KvretDatasetReader(DatasetReader):
    """
    A New Multi-Turn, Multi-Domain, Task-Oriented Dialogue Dataset.

    Stanford NLP released a corpus of 3,031 multi-turn dialogues in three distinct domains appropriate for an in-car assistant: calendar scheduling, weather information retrieval, and point-of-interest navigation. The dialogues are grounded through knowledge bases ensuring that they are versatile in their natural language without being completely free form.

    For details see https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/.
    """

    url = 'http://files.deeppavlov.ai/datasets/kvret_public.tar.gz'

    @staticmethod
    def _data_fname(datatype):
        assert datatype in ('train', 'dev', 'test'), "wrong datatype name"
        return 'kvret_{}_public.json'.format(datatype)

    @classmethod
    @overrides
    def read(self, data_path: str, dialogs: bool = False) -> Dict[str, List]:
        """
        Downloads ``'kvrest_public.tar.gz'``, decompresses, saves files to ``data_path``.

        Parameters:
            data_path: path to save data
            dialogs: flag indices whether to output list of turns or list of dialogs

        Returns:
            dictionary with ``'train'`` containing dialogs from ``'kvret_train_public.json'``, ``'valid'`` containing dialogs from ``'kvret_valid_public.json'``, ``'test'`` containing dialogs from ``'kvret_test_public.json'``. Each fields is a list of tuples ``(x_i, y_i)``.
        """

        required_files = (self._data_fname(dt) for dt in ('train', 'dev', 'test'))
        if not all(Path(data_path, f).exists() for f in required_files):
            log.info('[downloading dstc2 from {} to {}]'.format(self.url, data_path))
            download_decompress(self.url, data_path)
            mark_done(data_path)

        data = {
            'train': self._read_from_file(
                Path(data_path, self._data_fname('train')), dialogs),
            'valid': self._read_from_file(
                Path(data_path, self._data_fname('dev')), dialogs),
            'test': self._read_from_file(
                Path(data_path, self._data_fname('test')), dialogs)
        }
        return data

    @classmethod
    def _read_from_file(cls, file_path, dialogs=False):
        """Returns data from single file"""
        log.info("[loading dialogs from {}]".format(file_path))

        utterances, responses, dialog_indices = \
            cls._get_turns(cls._iter_file(file_path), with_indices=True)

        data = list(map(cls._format_turn, zip(utterances, responses)))

        if dialogs:
            return [data[idx['start']:idx['end']] for idx in dialog_indices]
        return data

    @staticmethod
    def _format_turn(turn):
        x = {'text': turn[0]['utterance'],
             'dialog_id': turn[0]['dialog_id'],
             'kb_columns': turn[0]['kb_columns'],
             'kb_items': turn[0]['kb_items'],
             'requested': turn[0].get('requested', {}),
             'slots': turn[0].get('slots', {})}
        if turn[0].get('episode_done') is not None:
            x['episode_done'] = turn[0]['episode_done']
        y = {'text': turn[1]['utterance'],
             'task': turn[0]['task'],
             'requested': turn[1].get('requested', {}),
             'slots': turn[1].get('slots', {})}
        return (x, y)

    @staticmethod
    def _check_dialog(dialog):
        # TODO: manually fix bad dialogs
        driver = True
        for turn in dialog:
            if turn['turn'] not in ('driver', 'assistant'):
                raise RuntimeError("Dataset wrong format: `turn` key value is"
                                   " either `driver` or `assistant`.")
            if driver and turn['turn'] != 'driver':
                log.debug("Turn is expected to by driver's, but it's {}'s" \
                          .format(turn['turn']))
                return False
            if not driver and turn['turn'] != 'assistant':
                log.debug("Turn is expected to be assistant's but it's {}'s" \
                          .format(turn['turn']))
                return False
            driver = not driver
        # if not driver:
        #    log.debug("Last turn is expected to be by assistant")
        #    return False
        return True

    @staticmethod
    def _filter_duplicates(dialog):
        last_turn, last_utter = None, None
        for turn in dialog:
            curr_turn, curr_utter = turn['turn'], turn['data']['utterance']
            if (curr_turn != last_turn) or (curr_utter != last_utter):
                yield turn
            last_turn, last_utter = curr_turn, curr_utter

    @classmethod
    def _iter_file(cls, file_path):
        with open(file_path, 'rt', encoding='utf8') as f:
            data = json.load(f)
        for i, sample in enumerate(data):
            dialog = list(cls._filter_duplicates(sample['dialogue']))
            if cls._check_dialog(dialog):
                yield dialog, sample['scenario']
            else:
                log.warning("Skipping {}th dialogue with uuid={}: wrong format." \
                            .format(i, sample['scenario']['uuid']))

    @staticmethod
    def _get_turns(data, with_indices=False):
        utterances, responses, dialog_indices = [], [], []
        for dialog, scenario in data:
            for i, turn in enumerate(dialog):
                replica = turn['data']
                if i == 0:
                    replica['episode_done'] = True
                if turn['turn'] == 'driver':
                    replica['task'] = scenario['task']
                    replica['dialog_id'] = scenario['uuid']
                    replica['kb_columns'] = scenario['kb']['column_names']
                    replica['kb_items'] = scenario['kb']['items']
                    utterances.append(replica)
                else:
                    responses.append(replica)

            # if last replica was by driver
            if len(responses) != len(utterances):
                utterances[-1]['end_dialogue'] = False
                responses.append({'utterance': '', 'end_dialogue': True})

            last_utter = responses[-1]['utterance']
            if last_utter and not last_utter[-1].isspace():
                last_utter += ' '
            responses[-1]['utterance'] = last_utter + 'END_OF_DIALOGUE'

            dialog_indices.append({
                'start': len(utterances),
                'end': len(utterances) + len(dialog),
            })

        if with_indices:
            return utterances, responses, dialog_indices
        return utterances, responses
