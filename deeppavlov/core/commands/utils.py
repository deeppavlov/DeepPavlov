"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from pathlib import Path

from typing import Union

from deeppavlov.core.common import paths
from deeppavlov.core.common.file import read_json


def set_deeppavlov_root(config_path: str):
    """
    Make a serialization user dir.
    """
    config = read_json(config_path)
    try:
        deeppavlov_root = Path(config['deeppavlov_root'])
    except KeyError:
        deeppavlov_root = Path(__file__, "..", "..", "..", "..", "download").resolve()

    deeppavlov_root.mkdir(exist_ok=True)

    paths.deeppavlov_root = deeppavlov_root


def get_deeppavlov_root() -> Path:
    return paths.deeppavlov_root


def expand_path(path: Union[str, Path]):
    return get_deeppavlov_root() / Path(path).expanduser()
