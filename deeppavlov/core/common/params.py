import sys

from typing import Dict, Type, TypeVar

from deeppavlov.core.common.registry import REGISTRY
from deeppavlov.core.common.errors import ConfigError

T = TypeVar('T')


def from_params(cls: Type, params: Dict, **kwargs) -> Type['T']:
    # what is passed in json:
    config_params = {k: v for k, v in params.items() if k != 'name'}

    # find the submodels params recursively
    for param_name, subcls_params in config_params.items():
        if isinstance(subcls_params, dict):
            try:
                subcls_name = subcls_params['name']
            except KeyError:
                "This parameter is passed as dict to the class constructor."
                " The user didn't intent it to be a model."
                continue
            try:
                subcls = REGISTRY[subcls_name]
            except KeyError:
                raise ConfigError(
                    "The class {} is not registered. Either register this class,"
                    " or rename the parameter.".format(subcls_name))
            config_params[param_name] = from_params(subcls, subcls_params,
                                                    vocabs=kwargs['vocabs'],
                                                    mode=kwargs['mode'])

    try:
        model = cls(**dict(config_params, **kwargs))
    except Exception:
        print("Exception in {}".format(cls), file=sys.stderr)
        raise

    return model
