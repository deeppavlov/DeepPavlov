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
import importlib
from typing import Dict, Type

from deeppavlov.core.commands.utils import expand_path, get_deeppavlov_root, set_deeppavlov_root
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import REGISTRY
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component

log = get_logger(__name__)

_refs = {}


def _resolve(val):
    if isinstance(val, str) and val.startswith('#'):
        component_id, *attributes = val[1:].split('.')
        try:
            val = _refs[component_id]
        except KeyError:
            e = ConfigError('Component with id "{id}" was referenced but not initialized'
                            .format(id=component_id))
            log.exception(e)
            raise e
        attributes = ['val'] + attributes
        val = eval('.'.join(attributes))
    return val


def from_params(params: Dict, **kwargs) -> Component:
    # what is passed in json:
    config_params = {k: _resolve(v) for k, v in params.items()}

    # get component by reference (if any)
    if 'ref' in config_params:
        try:
            return _refs[config_params['ref']]
        except KeyError:
            e = ConfigError('Component with id "{id}" was referenced but not initialized'
                            .format(id=config_params['ref']))
            log.exception(e)
            raise e

    elif 'config_path' in config_params:
        from deeppavlov.core.commands.infer import build_model_from_config
        deeppavlov_root = get_deeppavlov_root()
        config = read_json(expand_path(config_params['config_path']))
        model = build_model_from_config(config, as_component=True)
        set_deeppavlov_root({'deeppavlov_root': deeppavlov_root})
        return model

    elif 'class' in config_params:
        c = config_params.pop('class')
        try:
            module_name, cls_name = c.split(':')
            cls = getattr(importlib.import_module(module_name), cls_name)
        except ValueError:
            e = ConfigError('Expected class description in a `module.submodules:ClassName` form, but got `{}`'
                            .format(c))
            log.exception(e)
            raise e
    else:
        cls_name = config_params.pop('name', None)
        if not cls_name:
            e = ConfigError('Component config has no `name` nor `ref` or `class` fields')
            log.exception(e)
            raise e
        try:
            cls = REGISTRY[cls_name]
        except KeyError:
            e = ConfigError('Class {} is not registered.'.format(cls_name))
            log.exception(e)
            raise e

    # find the submodels params recursively
    for param_name, subcls_params in config_params.items():
        if isinstance(subcls_params, dict):
            if not {'ref', 'name', 'class', 'config_path'}.intersection(subcls_params):
                "This parameter is passed as dict to the class constructor."
                " The user didn't intent it to be a component."
                for k, v in subcls_params.items():
                    subcls_params[k] = _resolve(v)
                continue

            config_params[param_name] = from_params(subcls_params,
                                                    vocabs=kwargs['vocabs'],
                                                    mode=kwargs['mode'])

    try:
        component = cls(**dict(config_params, **kwargs))
        try:
            _refs[config_params['id']] = component
        except KeyError:
            pass
    except Exception:
        log.exception("Exception in {}".format(cls))
        raise

    return component
