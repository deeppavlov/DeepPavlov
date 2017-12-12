from typing import Dict, Type

from deeppavlov.common.registry import _REGISTRY


def from_params(cls: Type, params: Dict, **kwargs) -> Type:
    config_params = {}
    for sp in params.keys():
        try:
            if sp != 'name':
                config_params[sp] = params[sp]
        except KeyError as msg:
            print("Using default value for parameter `{}`.".format(sp))
            # Occurs when params[sp] throws KeyError. It means that the needed configuration is
            # absent in the json file and a default configuration from class constructor should
            #  be taken instead.
            pass

    for reg_name, subcl_params in config_params.items():
        if isinstance(subcl_params, dict):
            try:
                subcl = _REGISTRY[subcl_params['name']]
                subcl_params.pop('name')
                config_params[reg_name] = from_params(subcl, subcl_params)
            except KeyError as msg:
                # Occurs when v['name'] throws KeyError. Only those parameters that are registered
                # classes have 'name' keyword in their json config.
                raise KeyError(msg)
    return cls(**dict(config_params, **kwargs))
