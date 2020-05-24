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
    """
    The class to read the TweetQA dataset from files.

    Please, see https://tweetqa.github.io/.
    """

    def read(self, dir_path: Path, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Read TweetQA dataset from provided directory

        Args:
            dir_path: a path to a folder with dataset files
            url: link to archive with dataset, use url argument if non-default dataset is used

        Returns:
            dataset split on train/valid
        """
        if not dir_path.exists():
            dir_path.mkdir()

        dataset = {}
        for f, k in zip(['dev.json', 'train.json'], ['valid', 'train']):
            with dir_path.joinpath(f).open('r', encoding='utf8') as fp:
                data = json.load(fp)
            dataset[k] = data

        return dataset
