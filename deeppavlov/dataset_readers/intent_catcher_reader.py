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
from collections import defaultdict
from json import load as json_load
from yaml import full_load as yaml_load
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
        example = example[2:]
        if not regex:
            search_entities_re = re.compile(
                "\[[ a-zA-Z0-9]+\]\([ a-zA-Z0-9]+\)")
            search_entities = search_entities_re.search(example)
            while search_entities is not None:
                search_entities = search_entities_re.search(example)
                start, end = search_entities.span()
                example = example[:start] + re.sub("\]\([ a-zA-Z0-9]+\)", "", example[start:end])[
                    1:] + example[end:]
            example = re.sub("\?", "\?", example)
        return example

    def read(self, data_path: str, format: str = 'json', *args, **kwargs) -> Dict[str, List[Tuple[str, str]]]:
        data_types = ["train", "valid", "test"]

        train_file = kwargs.get('train', f'train.{format}')

        if not Path(data_path, train_file).exists():
            raise Exception(
                "data path {} does not exist or is empty!".format(
                    data_path))

        data = {"train": [],
                "valid": [],
                "test": []}

        if format == "yaml":  # load domain.yaml
            domain_file = Path(data_path, "domain.yml")
            if domain_file.exists():
                domain = [
                    self.parse_rasa_example(intent, regex=True)
                    for intent in yaml_load(open(domain_file))['intents'].split("\n")
                ]
            else:
                raise Exception(
                    "domain.yml in data path {} does not exist!".format(
                        data_path))

        for data_type in data_types:
            file_name = kwargs.get(
                data_type, '{}.{}'.format(data_type, format))
            if file_name is None:
                continue

            file = Path(data_path).joinpath(file_name)
            if file.exists():
                with open(file) as fp:
                    if format == 'json':
                        file = json_load(fp)
                    elif format == 'yaml':
                        file = yaml_load(fp)
                        file_data = defaultdict(list)
                        for part in file['nlu']:
                            if part.get('intent', '') in domain:
                                intent = part['intent']
                                regex = False
                            elif part.get('regex', '') in domain:
                                intent = part['regex']
                                regex = True
                            else:
                                continue
                            file_data[intent].extend([
                                self.parse_rasa_example(example, regex) for examples in part.get('examples', '').split("\n")
                            ])
                            if file['version'] == 'dp_2.0':
                                file_data[intent].extend([
                                    self.parse_rasa_example(example, True) for examples in part.get('regex_examples', '').split("\n")
                                ])
                        file = file_data
                for label in file:
                    data[data_type].extend([(phrase, label)
                                            for phrase in file[label]])
            else:
                log.warning("Cannot find {} file".format(file))

        return data
