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

from intent_recognition.utils import preprocessing
from deeppavlov.core.common.registry import register_model
import copy

@register_model('intent_preprocessing')
class IntentPreprocessing(object):
    """
    Class preprocesses data
    """
    @staticmethod
    def preprocess(dataset=None, data_type='train', data=None, *args, **kwargs):
        """
        Method preprocesses given data or field of the given dataset
        Args:
            dataset: dataset which field will be preprocessed
            data_type: field name that will be preprocessed
            data: optional, list of texts that will be preprocessed
            *args:
            **kwargs:

        Returns:
            preprocessed data
        """
        if data is not None:
            prep_data = preprocessing(data)
            return prep_data
        else:
            dataset_copy = copy.deepcopy(dataset)
            all_data = dataset.iter_all(data_type=data_type)
            texts = []
            for i, sample in enumerate(all_data):
                dataset_copy.data[data_type][i] = (preprocessing([sample[0]])[0], dataset.data[data_type][i][1])
            return dataset_copy
