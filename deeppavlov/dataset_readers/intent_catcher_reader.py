# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.file import read_yaml
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.dataset_readers.dto.rasa.nlu import IntentLine

log = getLogger(__file__)


@register('intent_catcher_reader')
class IntentCatcherReader(DatasetReader):
    """Reader for Intent Catcher dataset in json or YAML (RASA v2) format"""

    def parse_rasa_example(self, example: str, regex: bool = False):
        example = example[2:]
        if not regex:
            example = IntentLine.from_line(example).text
        return example

    def read(self, data_path: str, format: str = 'json', *args, **kwargs) -> Dict[str, List[Tuple[str, str]]]:
        data_types = ["train", "valid", "test"]

        if format == 'yaml':
            fmt = 'yml'
        elif format == 'json':
            fmt = 'json'
        else:
            raise Exception("Wrong file format. ")

        train_file = kwargs.get('train', f'train.{fmt}')

        if not Path(data_path, train_file).exists():
            raise Exception(
                "data path {} does not exist or is empty!".format(
                    data_path))

        data = {"train": [],
                "valid": [],
                "test": []}

        for data_type in data_types:
            file_name = kwargs.get(data_type, '{}.{}'.format(data_type, fmt))
            if file_name is None:
                continue

            file = Path(data_path).joinpath(file_name)
            if file.exists():
                ic_file_content = None
                if format == 'json':
                    ic_file_content = read_json(file)
                    raise Exception("json is not supported anymore."
                                    " Use RASA reader and YAML instead")

                elif format == 'yaml':
                    raise Exception("Use RASA reader instead")

                # noinspection PyUnboundLocalVariable
                data[data_type] = ic_file_content
            else:
                log.warning("Cannot find {} file".format(file))

        return data
