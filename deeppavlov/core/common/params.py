from typing import Dict, Type, TypeVar

from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.common.errors import ConfigError

T = TypeVar('T')


def from_params(cls: Type, params: Dict, **kwargs) -> Type['T']:
    config_params = {k: v for k, v in params.items() if k != 'name'}

    for param_name, subcl_params in config_params.items():
        if isinstance(subcl_params, dict):
            try:
                subcl_name = subcl_params['name']
            except KeyError:
                "This parameter is passed as dict to the class constructor."
                " The user didn't intent it to be a model."
                continue

            try:
                subcl = _REGISTRY[subcl_name]
                subcl_params.pop('name')
                config_params[param_name] = from_params(subcl, subcl_params)
            except KeyError:
                raise ConfigError(
                    "The class {} is not registered. Either register this class,"
                    " or rename the parameter.".format(
                        subcl_params['name']))
    # DEBUG
    # print(type(cls(**dict(config_params, **kwargs))))

    return cls(**dict(config_params, **kwargs))