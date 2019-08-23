# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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
from typing import Dict

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('faq_dict_reader')
class FaqDatasetReader(DatasetReader):
    """Reader for FAQ dataset"""

    def read(self, data_path: str, data: dict, **kwargs) -> Dict:
        """
        Read FAQ dataset from specified json file or remote url
        Parameters:
            data_path: path to json file of FAQ
            data: FAQ dictionary
        Returns:
            A dictionary containing training, validation and test parts of the dataset obtainable via
            ``train``, ``valid`` and ``test`` keys.
        """

        if data is None:
            raise ValueError("Please specify data parameter")

        xy_tuples = []

        for intent_name, faq_dict in data.items():
            for phrase in faq_dict['phrases']:
                xy_tuples.append((phrase.strip(), json.dumps({intent_name: faq_dict})))

        return {
            'train': xy_tuples,
            'valid': [],
            'test': []
        }
