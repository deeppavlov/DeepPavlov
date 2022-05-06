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
from copy import deepcopy
from pathlib import Path
from typing import Any, Union, Dict, TypeVar, Optional

from deeppavlov.core.common.file import read_json, find_config
from deeppavlov.core.common.registry import inverted_registry
from deeppavlov.core.data.utils import get_all_elems_from_json

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


def _update_requirements(config: dict) -> dict:
    """
    Generates requirements for DeepPavlov model and adds them as ``metadata.requirements`` field to the returned dict.

    Searches for the ``class_name`` keys in the passed config at all nesting levels. For each found component,
    function looks for dependencies in the requirements registry. Found dependencies are added to the returned copy of
    the config as ``metadata.requirements``. If the config already has ``metadata.requirements``, the existing one
    is complemented by the found requirements.

    Args:
        config: DeepPavlov model config
    Returns:
        config copy with updated ``metadata.requirements`` field according to the config components.
    """
    components = get_all_elems_from_json(config, 'class_name')
    components = {inverted_registry.get(component, component) for component in components}
    requirements_registry_path = Path(__file__).parents[1] / 'common' / 'requirements_registry.json'
    requirements_registry = read_json(requirements_registry_path)
    requirements = []
    for component in components:
        requirements.extend(requirements_registry.get(component, []))
    requirements.extend(config.get('metadata', {}).get('requirements', []))
    response = deepcopy(config)
    response['metadata'] = response.get('metadata', {})
    response['metadata']['requirements'] = list(set(requirements))
    return response


def _overwrite(data: Any, value: Any, nested_keys: list) -> None:
    """Changes ``data`` nested key value to ``value`` using ``nested_keys`` as nested keys list.

    Example:
        >>> x = {'a': [None, {'b': 2}]}
        >>> _overwrite(x, 42, ['a', 1, 'b'])
        >>> x
        {'a': [None, {'b': 42}]}

    """
    key = nested_keys.pop(0)
    if not nested_keys:
        data[key] = value
    else:
        _overwrite(data[key], value, nested_keys)


def parse_config(config: Union[str, Path, dict], overwrite: Optional[dict] = None) -> dict:
    """Apply metadata.variables values to placeholders inside config and update nested configs using overwrite parameter

    Args:
        config: Config to parse.
        overwrite: If not None - key-value pairs of nested keys and values to overwrite config.
            For {'chainer.pipe.0.class_name': 'simple_vocab'} it will update config
            config['chainer']['pipe'][0]['class_name'] = 'simple_vocab'.

    """
    if isinstance(config, (str, Path)):
        config = read_json(find_config(config))

    if overwrite is not None:
        for key, value in overwrite.items():
            items = [int(item) if item.isdigit() else item for item in key.split('.')]
            _overwrite(config, value, items)

    updated_config = _update_requirements(config)

    variables, variables_exact = _get_variables_from_config(updated_config)

    return _parse_config_property(updated_config, variables, variables_exact)


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
