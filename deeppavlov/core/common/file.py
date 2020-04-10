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
import pickle
from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import Union, Any

from ruamel.yaml import YAML

log = getLogger(__name__)


def find_config(pipeline_config_path: Union[str, Path]) -> Path:
    if not Path(pipeline_config_path).is_file():
        configs = [c for c in Path(__file__).parent.parent.parent.glob(f'configs/**/{pipeline_config_path}.json')
                   if str(c.with_suffix('')).endswith(pipeline_config_path)]  # a simple way to not allow * and ?
        if configs:
            log.info(f"Interpreting '{pipeline_config_path}' as '{configs[0]}'")
            pipeline_config_path = configs[0]
    return Path(pipeline_config_path)


def read_json(fpath: Union[str, Path]) -> dict:
    with open(fpath, encoding='utf8') as fin:
        return json.load(fin, object_pairs_hook=OrderedDict)


def save_json(data: dict, fpath: Union[str, Path]) -> None:
    with open(fpath, 'w', encoding='utf8') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)


def save_pickle(data: dict, fpath: Union[str, Path]) -> None:
    with open(fpath, 'wb') as fout:
        pickle.dump(data, fout, protocol=4)


def load_pickle(fpath: Union[str, Path]) -> Any:
    with open(fpath, 'rb') as fin:
        return pickle.load(fin)


def read_yaml(fpath: Union[str, Path]) -> dict:
    yaml = YAML(typ="safe")
    with open(fpath, encoding='utf8') as fin:
        return yaml.load(fin)
