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

import importlib
import json
from logging import getLogger
from pathlib import Path

from deeppavlov.core.common.errors import ConfigError

logger = getLogger(__name__)

_registry_path = Path(__file__).parent / 'registry.json'
if _registry_path.exists():
    with _registry_path.open(encoding='utf-8') as f:
        _REGISTRY = json.load(f)
else:
    _REGISTRY = {}

inverted_registry = {val: key for key, val in _REGISTRY.items()}


def cls_from_str(name: str) -> type:
    """Returns a class object with the name given as a string."""
    try:
        module_name, cls_name = name.split(':')
    except ValueError:
        raise ConfigError('Expected class description in a `module.submodules:ClassName` form, but got `{}`'
                          .format(name))

    return getattr(importlib.import_module(module_name), cls_name)


def register(name: str = None) -> type:
    """
    Register classes that could be initialized from JSON configuration file.
    If name is not passed, the class name is converted to snake-case.
    """

    def decorate(model_cls: type, reg_name: str = None) -> type:
        model_name = reg_name or short_name(model_cls)
        global _REGISTRY
        cls_name = model_cls.__module__ + ':' + model_cls.__name__
        if model_name in _REGISTRY and _REGISTRY[model_name] != cls_name:
            logger.warning('Registry name "{}" has been already registered and will be overwritten.'.format(model_name))
        _REGISTRY[model_name] = cls_name
        return model_cls

    return lambda model_cls_name: decorate(model_cls_name, name)


def short_name(cls: type) -> str:
    """Returns just a class name (without package and module specification)."""
    return cls.__name__.split('.')[-1]


def get_model(name: str) -> type:
    """Returns a registered class object with the name given in the string."""
    if name not in _REGISTRY:
        if ':' not in name:
            raise ConfigError("Model {} is not registered.".format(name))
        return cls_from_str(name)
    return cls_from_str(_REGISTRY[name])


def list_models() -> list:
    """Returns a list of names of registered classes."""
    return list(_REGISTRY)
