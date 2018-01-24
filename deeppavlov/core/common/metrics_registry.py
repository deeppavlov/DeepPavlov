import sys

from deeppavlov.core.common.errors import ConfigError

_REGISTRY = {}


def register_metric(metric_name):
    def decorate(f):
        if metric_name in _REGISTRY:
            print('"{}" is already registered as a metric name, the old function will be ignored'.format(metric_name),
                  file=sys.stderr)
        _REGISTRY[metric_name] = f
        return f
    return decorate


def get_metrics_by_names(names: list):
    not_found = [name for name in names if name not in _REGISTRY]
    if not_found:
        raise ConfigError('Names {} are not registered as metrics'.format(not_found))
    return [_REGISTRY[name] for name in names]
