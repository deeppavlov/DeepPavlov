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

import copy
import json
from pathlib import Path

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress, mark_done
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('dstc2_reader')
class DSTC2DatasetReader(DatasetReader):
    """
    Contains labelled dialogs from Dialog State Tracking Challenge 2
    http://camdial.org/~mh521/dstc/

    There were made the following modifications:
    1. new actions
        - bot dialog actions were concatenated into one action
        (example: {"dialog_acts": ["ask", "request"]} -> {"dialog_acts": ["ask_request"]})
        - if a slot key was associated with the dialog action, the new act
        was a concatenation of an act and a slot key
        (example: {"dialog_acts": ["ask"], "slot_vals": ["area"]} ->
        {"dialog_acts": ["ask_area"]})
    2. new train/dev/test split
        - original dstc2 consisted of three different MDP polices,
        the original train and dev datasets (consisting of two polices) were
        merged and randomly split into train/dev/test
    3. minor fixes
        - fixed several dialogs, where actions were wrongly annotated
        - uppercased first letter of bot responses
        - unified punctuation for bot responses'
    """

    url = 'http://lnsigo.mipt.ru/export/datasets/dstc2.tar.gz'

    @staticmethod
    def _data_fname(datatype):
        assert datatype in ('trn', 'val', 'tst'), "wrong datatype name"
        return 'dstc2-{}.jsonlist'.format(datatype)

    @overrides
    def read(self, data_path, dialogs=False):
    # TODO: mkdir if it doesn't exist

        required_files = (self._data_fname(dt) for dt in ('trn', 'val', 'tst'))
        if not all(Path(data_path, f).exists() for f in required_files):
            log.info('[downloading dstc2 from {} to {}]'.format(self.url, data_path))
            download_decompress(self.url, data_path)
            mark_done(data_path)

        data = {
            'train': self._read_from_file(
                Path(data_path, self._data_fname('trn')), dialogs),
            'valid': self._read_from_file(
                Path(data_path, self._data_fname('val')), dialogs),
            'test': self._read_from_file(
                Path(data_path, self._data_fname('tst')), dialogs)
        }
        return data

    @classmethod
    def _read_from_file(cls, file_path, dialogs=False):
        """Returns data from single file"""
        log.info("[loading dialogs from {}]".format(file_path))

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
        for ln in open(file_path, 'rt', encoding='utf8'):
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
        speaker1_turn = True
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
                speaker1_turn = True
            else:
                replica = _filter(turn)
                if speaker1_turn:
                    if episode_done:
                        replica['episode_done'] = True
                    utterances.append(replica)
                    num_dialog_utter += 1
                else:
                    responses.append(replica)
                    num_dialog_resp += 1
                episode_done = False
                speaker1_turn = not speaker1_turn

        if with_indices:
            return utterances, responses, dialog_indices
        return utterances, responses


@register('dstc2_v2_reader')
class DSTC2Version2DatasetReader(DatasetReader):
    """
    Contains labelled dialogs from Dialog State Tracking Challenge 2
    http://camdial.org/~mh521/dstc/

    There were made the following modifications:
    1. added api calls to restaurant database
        - example: api_call area="south" food="dontcare" pricerange="cheap"
    2. new actions
        - bot dialog actions were concatenated into one action
        (example: {"dialog_acts": ["ask", "request"]} -> {"dialog_acts": ["ask_request"]})
        - if a slot key was associated with the dialog action, the new act
        was a concatenation of an act and a slot key
        (example: {"dialog_acts": ["ask"], "slot_vals": ["area"]} ->
        {"dialog_acts": ["ask_area"]})
    3. new train/dev/test split
        - original dstc2 consisted of three different MDP polices,
        the original train and dev datasets (consisting of two polices) were
        merged and randomly split into train/dev/test
    4. minor fixes
        - fixed several dialogs, where actions were wrongly annotated
        - uppercased first letter of bot responses
        - unified punctuation for bot responses'
    """

    url = 'http://lnsigo.mipt.ru/export/datasets/dstc2_v2.tar.gz'

    @staticmethod
    def _data_fname(datatype):
        assert datatype in ('trn', 'val', 'tst'), "wrong datatype name"
        return 'dstc2-{}.jsonlist'.format(datatype)

    @overrides
    def read(self, data_path, dialogs=False):
        required_files = (self._data_fname(dt) for dt in ('trn', 'val', 'tst'))
        if not all(Path(data_path, f).exists() for f in required_files):
            log.info('[downloading data from {} to {}]'.format(self.url, data_path))
            download_decompress(self.url, data_path)
            mark_done(data_path)

        data = {
            'train': self._read_from_file(
                Path(data_path, self._data_fname('trn')), dialogs),
            'valid': self._read_from_file(
                Path(data_path, self._data_fname('val')), dialogs),
            'test': self._read_from_file(
                Path(data_path, self._data_fname('tst')), dialogs)
        }
        return data

    @classmethod
    def _read_from_file(cls, file_path, dialogs=False):
        """Returns data from single file"""
        log.info("[loading dialogs from {}]".format(file_path))

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
        for ln in open(file_path, 'rt', encoding='utf8'):
            if ln.strip():
                yield json.loads(ln)
            else:
                yield {}

    @staticmethod
    def _get_turns(data, with_indices=False):
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
                speaker = turn.pop('speaker')
                if speaker == 1:
                    if episode_done:
                        turn['episode_done'] = True
                    utterances.append(turn)
                    num_dialog_utter += 1
                elif speaker == 2:
                    if num_dialog_utter - 1 == num_dialog_resp:
                        responses.append(turn)
                    elif num_dialog_utter - 1 < num_dialog_resp:
                        if episode_done:
                            responses.append(turn)
                            utterances.append({
                                "text": "",
                                "dialog_acts": [],
                                "episode_done": True}
                            )
                        else:
                            new_turn = copy.deepcopy(utterances[-1])
                            if 'db_result' not in responses[-1]:
                                raise RuntimeError("Every api_call action should have"
                                                   " db_result, turn = {}"
                                                   .format(responses[-1]))
                            new_turn['db_result'] = responses[-1].pop('db_result')
                            utterances.append(new_turn)
                            responses.append(turn)
                        num_dialog_utter += 1
                    else:
                        raise RuntimeError("there cannot be two successive turns of"
                                           " speaker 1")
                    num_dialog_resp += 1
                else:
                    raise RuntimeError("Only speakers 1 and 2 are supported")
                episode_done = False

        if with_indices:
            return utterances, responses, dialog_indices
        return utterances, responses
