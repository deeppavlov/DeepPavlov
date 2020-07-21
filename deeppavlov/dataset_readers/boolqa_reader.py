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
from typing import Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress

@register('boolqa_reader')
class BoolqaReader(DatasetReader):
    """The class to read the boolqa dataset from files.
    """

    urls = { 
            'en': 'http://files.deeppavlov.ai/datasets/BoolQ.tar.gz',
            'ru': 'http://files.deeppavlov.ai/datasets/DaNetQA.tar.gz'
           }

    def read(self,
             data_path: str,
             language: str = 'en',
             *args, **kwargs) -> Dict[str, List[Tuple[Tuple[str, str], int]]]:

        """Read the BoolQ dataset from files.

        Args:
            data_path: A path to a folder with dataset files.
            language: The dataset language ('ru', 'en' are available)
        """

        if language in self.urls:
            self.url = self.urls[language]
        else:
             raise RuntimeError(f'The dataset for {language} is unavailable')

        data_path = expand_path(data_path)
        if not data_path.exists():
            dir_path.mkdir(parents=True)

        download_decompress(self.url, data_path)
        dataset = {}

        for filename in ['train.jsonl', 'valid.jsonl']:
           dataset[filename.split('.')[0]] = self._build_data(language, data_path / filename)

        return dataset

    @staticmethod
    def _build_data(ln: str, data_path: Path) -> List[Tuple[Tuple[str, str], int]]:

        data = {}
        with open(data_path, 'r') as f:
            for line in f:
                jline = json.loads(line)
                if ln == 'ru':
                    if 'label' in jline:
                        data[tuple([jline['question'], jline['passage']])] = int(jline['label'])
                if ln == 'en':
                    if 'answer' in jline:
                        data[tuple([jline['question'], jline['passage']])] = int(jline['answer'])

        return list(data.items())
