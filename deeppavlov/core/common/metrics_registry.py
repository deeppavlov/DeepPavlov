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
from typing import Callable, Any

from deeppavlov.core.common.errors import ConfigError

log = getLogger(__name__)

_registry_path = Path(__file__).parent / 'metrics_registry.json'
if _registry_path.exists():
    with _registry_path.open(encoding='utf-8') as f:
        _REGISTRY = json.load(f)
else:
    _REGISTRY = {}


def fn_from_str(name: str) -> Callable[..., Any]:
    """Returns a function object with the name given in string."""
    try:
        module_name, fn_name = name.split(':')
        return getattr(importlib.import_module(module_name), fn_name)
    except ValueError:
        raise ConfigError('Expected function description in a `module.submodules:function_name` form, but got `{}`'
                          .format(name))
    except AttributeError:
        # noinspection PyUnboundLocalVariable
        raise ConfigError(f"Incorrect metric: '{module_name}' has no attribute '{fn_name}'.")


def register_metric(metric_name: str) -> Callable[..., Any]:
    """Decorator for metric registration."""

    def decorate(fn):
        fn_name = fn.__module__ + ':' + fn.__name__
        if metric_name in _REGISTRY and _REGISTRY[metric_name] != fn_name:
            log.warning('"{}" is already registered as a metric name, the old function will be ignored'
                        .format(metric_name))
        _REGISTRY[metric_name] = fn_name
        return fn

    return decorate


def get_metric_by_name(name: str) -> Callable[..., Any]:
    """Returns a metric callable with a corresponding name."""
    name = _REGISTRY.get(name, name)
    return fn_from_str(name)
