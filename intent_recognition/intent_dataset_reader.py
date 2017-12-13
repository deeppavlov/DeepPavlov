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

import os
import pandas as pd
from deeppavlov.core.common.registry import register_model
from deeppavlov.data.dataset_readers.dataset_reader import DatasetReader

@register_model('intent_dataset_reader')
class IntentDatasetReader(DatasetReader):
    """
    IntentDatasetReader reads data from some location and constructs a dict of given datasets.
    """
    @staticmethod
    def read(train_data_path=None, valid_data_path=None, test_data_path=None, *args, **kwargs):
        """
        Read a file from a path and returns data as dict with given datasets.
        """
        data_dict = dict()

        if train_data_path is not None:
            print('___Reading train data from %s' % train_data_path)
            if os.path.isfile(train_data_path):
                train_data = pd.read_csv(train_data_path)
                samples = []
                for i in range(train_data.shape[0]):
                    samples.append((train_data.loc[i,'text'], train_data.loc[i, 'intents'].split(' ')))
                data_dict['train'] = samples
            else:
                raise IOError("Error: Train file does not exist")
        if valid_data_path is not None:
            print('___Reading valid data from %s' % train_data_path)
            if os.path.isfile(valid_data_path):
                valid_data = pd.read_csv(valid_data_path)
                samples = []
                for i in range(valid_data.shape[0]):
                    samples.append((valid_data.loc[i, 'text'], valid_data.loc[i, 'intents'].split(' ')))
                data_dict['valid'] = samples
            else:
                raise IOError("Error: Valid file does not exist")
        if test_data_path is not None:
            print('___Reading test data from %s' % train_data_path)
            if os.path.isfile(test_data_path):
                test_data = pd.read_csv(test_data_path)
                samples = []
                for i in range(test_data.shape[0]):
                    samples.append((test_data.loc[i, 'text'], test_data.loc[i, 'intents'].split(' ')))
                data_dict['test'] = samples
            else:
                raise IOError("Error: Test file does not exist")

        return data_dict
