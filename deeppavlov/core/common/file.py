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
from typing import Union, Any, Iterable

from deeppavlov.core.common.aliases import ALIASES

log = getLogger(__name__)

_red_text, _reset_text_color, _sharp_line = "\x1b[31;20m", "\x1b[0m", '#'*80
DEPRECATOIN_MSG = f"{_red_text}\n\n{_sharp_line}\n" \
                  "# The model '{0}' has been removed from the DeepPavlov configs.\n" \
                  "# The model '{1}' is used instead.\n" \
                  "# To disable this message please switch to '{1}'.\n" \
                  "# Automatic name resolving will be disabled in the deeppavlov 1.2.0,\n" \
                  "# and if you try to use '{0}' you will get an ERROR.\n" \
                  f"{_sharp_line}{_reset_text_color}\n"


def find_config(pipeline_config_path: Union[str, Path]) -> Path:
    if pipeline_config_path in ALIASES:
        new_pipeline_config_path = ALIASES[pipeline_config_path]
        log.warning(DEPRECATOIN_MSG.format(pipeline_config_path, new_pipeline_config_path))
        pipeline_config_path = new_pipeline_config_path

    if not Path(pipeline_config_path).is_file():
        configs = [c for c in Path(__file__).parent.parent.parent.glob(f'configs/**/{pipeline_config_path}.json')
                   if str(c.with_suffix('')).endswith(pipeline_config_path)]  # a simple way to not allow * and ?
        if configs:
            log.debug(f"Interpreting '{pipeline_config_path}' as '{configs[0]}'")
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


def save_jsonl(data: Iterable[dict], fpath: Union[str, Path]) -> None:
    with open(fpath, 'w') as f:
        for item in data:
            f.write(f"{json.dumps(item, ensure_ascii=False)}\n")
