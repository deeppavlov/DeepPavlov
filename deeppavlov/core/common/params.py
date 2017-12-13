from inspect import getfullargspec

from typing import Dict, Type

from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.common.errors import ConfigError


def from_params(cls: Type, params: Dict, **kwargs) -> Type:
    signature_params = getfullargspec(cls.__init__).args[1:]
    config_params = {}
    for sp in signature_params:
        try:
            config_params[sp] = params[sp]
        except KeyError:
            # Occurs when params[sp] throws KeyError. It means that the needed configuration is
            # absent in the json file and a default configuration from class constructor should
            # be taken instead.
            pass

    for reg_name, subcl_params in config_params.items():
        if isinstance(subcl_params, dict):
            try:
                subcl = _REGISTRY[subcl_params['name']]
                subcl_params.pop('name')
                config_params[reg_name] = from_params(subcl, subcl_params)
            except KeyError:
                # raise ConfigError(
                #     "Your registered classes should have 'name' keyword in the config.")
                pass
    return cls(**dict(config_params, **kwargs))

