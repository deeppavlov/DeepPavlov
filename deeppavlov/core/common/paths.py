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
import shutil
from pathlib import Path


deeppavlov_root = ''
root_path = Path(__file__).resolve().parent.parent.parent.parent
default_settings_path = root_path / 'utils' / 'settings' / '.default'


def get_settings_path() -> Path:
    """Returns DeepPavlov settings folder absolute path."""
    with open(root_path / 'deeppavlov/paths.json', encoding='utf8') as f:
        paths = json.load(f)
    settings_paths = Path(paths['settings_path']).resolve() if paths['settings_path'][0] == '/' \
        else root_path / paths['settings_path']
    return settings_paths


def set_settings_path(settings_path: Path):
    """Sets new DeepPavlov settings folder and moves settings files from old one.

    Args:
        settings_path: New settings path.
    """
    with open(root_path / 'deeppavlov/paths.json', encoding='utf8') as f:
        paths = json.load(f)
    old_settings_path = Path(paths['settings_path']).resolve()

    if settings_path == old_settings_path:
        print('New settings path is equal to current settings path')
    else:
        if settings_path.is_dir():
            confirmation_message = 'Specified folder exists. Overwrite files in it? (y[es]/n[o]): '
            confirm = input(confirmation_message)
            confirm = True if str(confirm).lower() in {'y', 'yes'} else False
        else:
            settings_path.mkdir(parents=True)
            confirm = True

        if confirm:
            for file in [file.name for file in default_settings_path.iterdir()]:
                old_file = old_settings_path / file
                shutil.copy(old_file, settings_path / file)
                old_file.unlink()

            paths['settings_path'] = str(settings_path)
            with open(root_path / 'deeppavlov/paths.json', 'w', encoding='utf8') as f:
                json.dump(paths, f, ensure_ascii=False, indent=2)

            print(f'New settings path was set and all settings files were moved to: {str(settings_path)}')


def set_settings_default():
    """Sets ALL settings files and settings directory to DeepPavlov defaults."""
    old_settings_path = get_settings_path()
    with open(root_path / 'deeppavlov/paths.json', encoding='utf8') as f:
        paths = json.load(f)
    paths['settings_path'] = 'utils/settings'
    with open(root_path / 'deeppavlov/paths.json', 'w', encoding='utf8') as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)
    new_settings_path = get_settings_path()

    for settings_file in default_settings_path.iterdir():
        old_file = old_settings_path / settings_file.name
        old_file.unlink()
        shutil.copy(settings_file, new_settings_path / settings_file.name)

    print('All DeepPavlov settings were set to default')
