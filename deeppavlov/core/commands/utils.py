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

from pathlib import Path, PosixPath

from deeppavlov.core.common import paths
from deeppavlov.core.common.file import read_json


def set_usr_dir(config_path: str, usr_dir_name='download'):
    """
    Make a serialization user dir.
    """
    config = read_json(config_path)
    try:
        usr_dir = Path(config['usr_dir'])
    except KeyError:
        root_dir = (Path(__file__) / ".." / ".." / ".." / "..").resolve()
        usr_dir = root_dir / usr_dir_name

    usr_dir.mkdir(exist_ok=True)

    paths.USR_PATH = usr_dir


def get_usr_dir() -> PosixPath:
    return paths.USR_PATH
