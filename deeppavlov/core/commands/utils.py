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

from pathlib import Path
from typing import Union
import os

from deeppavlov.core.common import paths


def set_deeppavlov_root(config: dict) -> None:
    """Make a serialization user dir."""
    try:
        deeppavlov_root = Path(config['deeppavlov_root'])
    except KeyError:
        deeppavlov_root = Path(__file__, "..", "..", "..", "..", "download").resolve()

    deeppavlov_root.mkdir(exist_ok=True)

    paths.deeppavlov_root = deeppavlov_root


def get_deeppavlov_root() -> Path:
    """Return DeepPavlov root directory."""
    if not paths.deeppavlov_root:
        set_deeppavlov_root({})
    return paths.deeppavlov_root


def expand_path(path: Union[str, Path]) -> Path:
    """Make path expansion."""
    return get_deeppavlov_root() / Path(path).expanduser()


def make_all_dirs(path: Union[str, Path]) -> None:
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def is_file_exist(path: Union[str, Path]):
    if path is None:
        return False

    return os.path.exists(expand_path(path))


def is_empty(d: Path) -> bool:
    """Check if directory is empty."""
    return not bool(list(d.iterdir()))


def import_packages(packages: list) -> None:
    """Simple function to import packages from list."""
    for package in packages:
        __import__(package)
