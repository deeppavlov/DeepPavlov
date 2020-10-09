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

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('rel_ranking_reader')
class ParaphraserReader(DatasetReader):
    """The class to read the paraphraser.ru dataset from files.
​
    Please, see https://paraphraser.ru.
    """

    def read(self,
             data_path: str,
             do_lower_case: bool = True,
             *args, **kwargs) -> Dict[str, List[Tuple[Tuple[str, str], int]]]:
        """Read the paraphraser.ru dataset from files.
​
        Args:
            data_path: A path to a folder with dataset files.
            do_lower_case: Do you want to lowercase all texts
        """

        data_path = expand_path(data_path)
        train_fname = data_path / 'paraphrases.xml'
        test_fname = data_path / 'paraphrases_gold.xml'

        train_data = self._build_data(train_fname, do_lower_case)
        test_data = self._build_data(test_fname, do_lower_case)
        return {"train": train_data, "valid": [], "test": test_data}

    @staticmethod
    def _build_data(data_path: Path, do_lower_case: bool) -> List[Tuple[Tuple[str, str], int]]:
        root = ET.fromstring(data_path.read_text(encoding='utf8'))
        data = []
        for paraphrase in root.findall('corpus/paraphrase'):
            key = (paraphrase.find('value[@name="text_1"]').text,
                   paraphrase.find('value[@name="text_2"]').text)
            if do_lower_case:
                key = tuple([t.lower() for t in key])

            pos_or_neg = int(paraphrase.find('value[@name="class"]').text)
            data.append((key, pos_or_neg))
        return data
