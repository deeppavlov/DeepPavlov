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

log = getLogger(__file__)


@register('intent_catcher_reader')
class IntentCatcherReader(DatasetReader):
    """Reader for Intent Catcher dataset in json or YAML (RASA v2) format"""

    def parse_rasa_example(self, example: str, regex: bool = False):
        search_entities_re = re.compile(
            "\[[ a-zA-Z0-9]+\]\([ a-zA-Z0-9]+\)")
        example = example[2:]
        if not regex:
            search_entities = search_entities_re.search(example)
            while search_entities is not None:
                start, end = search_entities.span()
                example = example[:start] + re.sub("\]\([ a-zA-Z0-9]+\)", "", example[start:end])[
                    1:] + example[end:]
                search_entities = search_entities_re.search(example)
            example = re.sub("\?", "\?", example)
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
                if format == 'json':
                    ic_file_content = read_json(file)
                elif format == 'yaml':
                    domain_file = Path(data_path, "domain.yml")
                    if domain_file.exists():
                        domain = read_yaml(domain_file)['intents']
                    else:
                        raise Exception("domain.yml in data path {} does not exist!".format(data_path))

                    ic_file_content = read_yaml(file)
                    file_data = defaultdict(list)
                    for part in ic_file_content['nlu']:
                        if part.get('intent', '') in domain:
                            intent = part['intent']
                            regex = False
                        elif part.get('regex', '') in domain:
                            intent = part['regex']
                            regex = True
                        else:
                            continue
                        file_data[intent].extend([
                                self.parse_rasa_example(example, regex) for example in part.get('examples', '').split("\n")
                        ])
                        if file['version'] == 'dp_2.0':
                            file_data[intent].extend([self.parse_rasa_example(example, True) for example in part.get('regex_examples', '').split("\n")])
                    ic_file_content = file_data

                # noinspection PyUnboundLocalVariable
                data[data_type] = ic_file_content
            else:
                log.warning("Cannot find {} file".format(file))

        return data
