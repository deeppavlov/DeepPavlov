"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys

from typing import Dict, Type, TypeVar

from deeppavlov.core.common.registry import REGISTRY
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

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
                    " or rename the parameter.".format(
                        subcls_params['name']))
            config_params[param_name] = from_params(subcls, subcls_params,
                                                    vocabs=kwargs['vocabs'],
                                                    mode=kwargs['mode'])

    try:
        model = cls(**dict(config_params, **kwargs))
    except Exception:
        log.error("Exception in {}".format(cls), exc_info=True)
        raise

    return model
