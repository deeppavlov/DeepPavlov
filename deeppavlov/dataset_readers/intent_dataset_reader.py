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


import json
import pandas as pd
from pathlib import Path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('intent_dataset_reader')
class IntentDatasetReader(DatasetReader):
    """
    IntentDatasetReader reads data from some location and constructs a dict of given datasets.
    """
    @staticmethod
    def read(data_path=None, *args, **kwargs):
        """
        Read a file from a path and returns data as dict with given datasets.
        """
        data_dict = dict()
        data_path = Path(data_path)
        train_data_path = data_path / "dstc2-trn.jsonlist"
        valid_data_path = data_path / "dstc2-val.jsonlist"
        test_data_path = data_path / "dstc2-tst.jsonlist"

        if Path(train_data_path).is_file():
            print('___Reading train data from {}'.format(train_data_path))
            data_dict['train'] = IntentDatasetReader.read_from_json(train_data_path)
        else:
            raise IOError("Error: Train file does not exist")

        if Path(valid_data_path).is_file():
            print('___Reading valid data from {}'.format(train_data_path))
            data_dict['valid'] = IntentDatasetReader.read_from_json(valid_data_path)
        else:
            raise IOError("Error: Valid file does not exist")

        if Path(test_data_path).is_file():
            print('___Reading test data from {}'.format(train_data_path))
            data_dict['test'] = IntentDatasetReader.read_from_json(test_data_path)
        else:
            raise IOError("Error: Test file does not exist")

        return data_dict

    @staticmethod
    def read_from_json(data_file):
        data = []
        with open(data_file) as read:
            for line in read:
                line = line.strip()
                # if empty line - it is the end of dialog
                if not line:
                    continue

                replica = json.loads(line)
                if 'goals' not in replica.keys():
                    # bot reply
                    continue
                curr_intents = []
                if replica['dialog_acts']:
                    for act in replica['dialog_acts']:
                        for slot in act['slots']:
                            if slot[0] == 'slot':
                                curr_intents.append(act['act'] + '_' + slot[1])
                            else:
                                curr_intents.append(act['act'] + '_' + slot[0])
                        if len(act['slots']) == 0:
                            curr_intents.append(act['act'])
                else:
                    if replica['text']:
                        curr_intents.append('unknown')
                    else:
                        continue
                data.append({'text': replica['text'],
                             'intents': ' '.join(curr_intents)})
        data = pd.DataFrame(data)
        samples = []
        for i in range(data.shape[0]):
            samples.append((data.loc[i, 'text'], data.loc[i, 'intents'].split(' ')))
        return samples
