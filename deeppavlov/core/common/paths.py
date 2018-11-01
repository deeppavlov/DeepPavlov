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
default_configs_path = root_path / 'utils' / 'configs' / '.default'


def get_configs_path() -> Path:
    """Returns DeepPavlov configs folder absolute path."""
    with open(root_path / 'deeppavlov/paths.json', encoding='utf8') as f:
        paths = json.load(f)
    configs_paths = Path(paths['configs_path']).resolve() if paths['configs_path'][0] == '/' \
        else root_path / paths['configs_path']
    return configs_paths


def set_configs_path(configs_path: Path):
    """Sets new DeepPavlov configs folder and moves configs from old one.

    Args:
        configs_path: New config path.
    """
    with open(root_path / 'deeppavlov/paths.json', encoding='utf8') as f:
        paths = json.load(f)
    old_configs_path = Path(paths['configs_path']).resolve()

    if configs_path == old_configs_path:
        print('New configs path is equal to current configs path')
    else:
        if configs_path.is_dir():
            confirmation_message = 'Specified folder exists. Overwrite files in it? (y[es]/n[o]): '
            confirm = input(confirmation_message)
            confirm = True if str(confirm).lower() in {'y', 'yes'} else False
        else:
            configs_path.mkdir(parents=True)
            confirm = True

        if confirm:
            for file in [file.name for file in default_configs_path.iterdir()]:
                old_file = old_configs_path / file
                shutil.copy(old_file, configs_path / file)
                old_file.unlink()

            paths['configs_path'] = str(configs_path)

            with open(root_path / 'deeppavlov/paths.json', 'w', encoding='utf8') as f:
                json.dump(paths, f, ensure_ascii=False, indent=2)
            print(f'New configs path was set and all config files were moved to: {str(configs_path)}')


def set_configs_default():
    """Sets ALL config files and config directory to DeepPavlov defaults."""
    old_configs_path = get_configs_path()
    with open(root_path / 'deeppavlov/paths.json', encoding='utf8') as f:
        paths = json.load(f)
    paths['configs_path'] = 'utils/configs'
    with open(root_path / 'deeppavlov/paths.json', 'w', encoding='utf8') as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)
    new_configs_path = get_configs_path()

    for config_file in default_configs_path.iterdir():
        old_file = old_configs_path / config_file.name
        old_file.unlink()
        shutil.copy(config_file, new_configs_path / config_file.name)

    print('All DeepPavlov configs were set to default')
