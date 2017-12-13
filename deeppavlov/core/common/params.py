from typing import Dict, Type, TypeVar

from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.common.errors import ConfigError

T = TypeVar('T')


def from_params(cls: Type, params: Dict, **kwargs) -> Type['T']:
    config_params = {}
    for sp in params.keys():
        try:
            if sp != 'name':
                config_params[sp] = params[sp]
        except KeyError:
            print("Using default value for parameter `{}`.".format(sp))
            # Occurs when params[sp] throws KeyError. It means that the needed configuration is
            # absent in the json file and a default configuration from class constructor should
            #  be taken instead.
            pass

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

    return cls(**dict(config_params, **kwargs))
