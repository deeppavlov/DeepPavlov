from typing import Dict, Type, TypeVar

from deeppavlov.core.common.registry import REGISTRY
from deeppavlov.core.common.errors import ConfigError

T = TypeVar('T')


def from_params(cls: Type, params: Dict, vocabs: Dict=dict(), **kwargs) -> Type['T']:

    # what is passed in json:
    config_params = {k: v for k, v in params.items() if k not in {'name', 'vocabs'}}

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
                subcls_params.pop('name')
                config_params[param_name] = from_params(subcls, subcls_params, vocabs)
            except KeyError:
                raise ConfigError(
                    "The class {} is not registered. Either register this class,"
                    " or rename the parameter.".format(
                        subcls_params['name']))

    final_params = {k: v for k, v in config_params.items()}

    # set superclass parameters:
    for super_attr, v in config_params.items():
        if hasattr(cls, super_attr):
            setattr(cls, super_attr, v)
            final_params.pop(super_attr)

    # return an instance
    if 'vocabs' in params:
        input_vocabs = {key: vocabs[key] for key in params['vocabs']}
        model = cls(**dict(final_params, **input_vocabs, **kwargs))
    else:
        model = cls(**dict(final_params, **kwargs))
    return model
