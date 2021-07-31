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

from json import load
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader

log = getLogger(__file__)


@register('intent_catcher_reader')
class IntentCatcherReader(DatasetReader):
    """Reader for Intent Catcher dataset in json format"""

    def read(self, data_path: str, *args, **kwargs) -> Dict[str, List[Tuple[str, str]]]:
        data_types = ["train", "valid", "test"]

        train_file = kwargs.get('train', 'train.json')

        if not Path(data_path, train_file).exists():
            raise Exception(
                "data path {} does not exist or is empty!".format(
                    data_path))

        data = {"train": [],
                "valid": [],
                "test": []}

        for data_type in data_types:
            file_name = kwargs.get(data_type, '{}.{}'.format(data_type, "json"))
            if file_name is None:
                continue

            file = Path(data_path).joinpath(file_name)
            if file.exists():
                with open(file) as fp:
                    file = load(fp)
                for label in file:
                    data[data_type].extend([(phrase, label) for phrase in file[label]])
            else:
                log.warning("Cannot find {} file".format(file))

        return data
