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
import os
import shutil

from pathlib import Path

_root_path = Path(__file__).resolve().parents[3]
_default_settings_path: Path = _root_path / 'deeppavlov' / 'utils' / 'settings'
_settings_path = Path(os.getenv('DP_SETTINGS_PATH', _default_settings_path)).expanduser().resolve()
if _settings_path.is_file():
    raise FileExistsError(f'DP_SETTINGS_PATH={_settings_path} is a file and not a directory')

if _default_settings_path in _settings_path.parents:
    raise RecursionError(f'DP_SETTINGS_PATH={_settings_path} is relative'
                         f' to the default settings path {_default_settings_path}')


def get_settings_path() -> Path:
    """Return an absolute path to the DeepPavlov settings directory"""
    populate_settings_dir()
    return _settings_path


def populate_settings_dir(force: bool = False) -> bool:
    """
    Populate settings directory with default settings files

    Args:
        force: if ``True``, replace existing settings files with default ones

    Returns:
        ``True`` if any files were copied and ``False`` otherwise
    """
    res = False
    if _default_settings_path == _settings_path:
        return res

    for src in list(_default_settings_path.glob('**/*.json')):
        dest = _settings_path / src.relative_to(_default_settings_path)
        if not force and dest.exists():
            continue
        res = True
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
    return res
