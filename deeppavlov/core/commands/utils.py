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
from pathlib import Path
from typing import Union, Dict, TypeVar

from deeppavlov.core.common.file import read_json, find_config

# noinspection PyShadowingBuiltins
_T = TypeVar('_T', str, float, bool, list, dict)


def _parse_config_property(item: _T, variables: Dict[str, Union[str, Path, float, bool, int, None]],
                           variables_exact: Dict[str, Union[str, Path, float, bool, int, None]]) -> _T:
    """Recursively apply config's variables values to its property"""
    if isinstance(item, str):
        if item in variables_exact:
            return variables_exact[item]
        return item.format(**variables)
    elif isinstance(item, list):
        return [_parse_config_property(item, variables, variables_exact) for item in item]
    elif isinstance(item, dict):
        return {k: _parse_config_property(v, variables, variables_exact) for k, v in item.items()}
    else:
        return item


def _get_variables_from_config(config: Union[str, Path, dict]):
    """Read config's variables"""
    if isinstance(config, (str, Path)):
        config = read_json(find_config(config))

    variables = {
        'DEEPPAVLOV_PATH': os.getenv(f'DP_DEEPPAVLOV_PATH', Path(__file__).parent.parent.parent)
    }
    variables_exact = {f'{{{k}}}': v for k, v in variables.items()}
    for name, value in config.get('metadata', {}).get('variables', {}).items():
        env_name = f'DP_{name}'
        if env_name in os.environ:
            value = os.getenv(env_name)
        if value in variables_exact:
            value = variables_exact[value]
        elif isinstance(value, str):
            value = value.format(**variables)
        variables[name] = value
        variables_exact[f'{{{name}}}'] = value

    return variables, variables_exact


def parse_config(config: Union[str, Path, dict]) -> dict:
    """Apply variables' values to all its properties"""
    if isinstance(config, (str, Path)):
        config = read_json(find_config(config))

    variables, variables_exact = _get_variables_from_config(config)

    return _parse_config_property(config, variables, variables_exact)


def expand_path(path: Union[str, Path]) -> Path:
    """Convert relative paths to absolute with resolving user directory."""
    return Path(path).expanduser().resolve()


def import_packages(packages: list) -> None:
    """Import packages from list to execute their code."""
    for package in packages:
        __import__(package)


def parse_value_with_config(value: Union[str, Path], config: Union[str, Path, dict]) -> Path:
    """Fill the variables in `value` with variables values from `config`.
    `value` should be a string. If `value` is a string of only variable, `value` will be replaced with
    variable's value from config (the variable's value could be anything then)."""
    variables, variables_exact = _get_variables_from_config(config)

    return _parse_config_property(str(value), variables, variables_exact)
