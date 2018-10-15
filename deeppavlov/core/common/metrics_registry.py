import importlib
import json
from pathlib import Path
from typing import Callable, Any

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)

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
    except ValueError:
        raise ConfigError('Expected function description in a `module.submodules:function_name` form, but got `{}`'
                          .format(name))

    return getattr(importlib.import_module(module_name), fn_name)


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
    if name not in _REGISTRY:
        raise ConfigError(f'"{name}" is not registered as a metric')
    return fn_from_str(_REGISTRY[name])
