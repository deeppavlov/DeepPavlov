from typing import Dict, Type, TypeVar

from deeppavlov.core.common.registry import REGISTRY
from deeppavlov.core.common.errors import ConfigError

T = TypeVar('T')


def from_params(cls: Type, params: Dict, **kwargs) -> Type['T']:

    # what is passed in json:
    config_params = {k: v for k, v in params.items() if k != 'name'}

    # find the submodels params recursively
    for param_name, subcl_params in config_params.items():
        if isinstance(subcl_params, dict):
            try:
                subcl_name = subcl_params['name']
            except KeyError:
                "This parameter is passed as dict to the class constructor."
                " The user didn't intent it to be a model."
                continue

            try:
                subcl = REGISTRY[subcl_name]
                subcl_params.pop('name')
                config_params[param_name] = from_params(subcl, subcl_params)
            except KeyError:
                raise ConfigError(
                    "The class {} is not registered. Either register this class,"
                    " or rename the parameter.".format(
                        subcl_params['name']))

    final_params = {k: v for k, v in config_params.items()}

    # set superclass parameters:
    for super_attr, v in config_params.items():
        if hasattr(cls, super_attr):
            setattr(cls, super_attr, v)
            final_params.pop(super_attr)

    # return an instance
    return cls(**dict(final_params, **kwargs))
