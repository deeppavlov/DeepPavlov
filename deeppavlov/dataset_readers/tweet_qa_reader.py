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
from pathlib import Path
from typing import Dict, Any, Optional

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress


@register('tweetqa_dataset_reader')
class TweetQADatasetReader(DatasetReader):

    def read(self, dir_path: str, dataset: Optional[str] = 'SQuAD', url: Optional[str] = None, *args, **kwargs) \
            -> Dict[str, Dict[str, Any]]:
        """

        Args:
            dir_path: path to save data
            dataset: default dataset names: ``'SQuAD'``, ``'SberSQuAD'`` or ``'MultiSQuAD'``
            url: link to archive with dataset, use url argument if non-default dataset is used

        Returns:
            dataset split on train/valid

        Raises:
            RuntimeError: if `dataset` is not one of these: ``'SQuAD'``, ``'SberSQuAD'``, ``'MultiSQuAD'``.
        """
        
        
#         required_files = ['dev.json', 'test.json', 'train.json']
        required_files = ['dev.json', 'train.json']
        if not dir_path.exists():
            dir_path.mkdir()

#         if not all((dir_path / f).exists() for f in required_files):
#             download_decompress(self.url, dir_path)

        dataset = {}
        for f in required_files:
            with dir_path.joinpath(f).open('r', encoding='utf8') as fp:
                data = json.load(fp)
            if f == 'dev.json':
                dataset['valid'] = data
            elif f == 'train.json':
                dataset['train'] = data
        
        print('valid', len(dataset['valid']))
        print('train', len(dataset['train']))
        return dataset
