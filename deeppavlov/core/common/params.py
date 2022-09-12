# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from logging import getLogger
from types import FunctionType
from typing import Any, Dict, Union

from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import get_model
from deeppavlov.core.models.component import Component

log = getLogger(__name__)

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


def _init_param(param, mode):
    if isinstance(param, str):
        param = _resolve(param)
    elif isinstance(param, (list, tuple)):
        param = [_init_param(p, mode) for p in param]
    elif isinstance(param, dict):
        if {'ref', 'class_name', 'config_path'}.intersection(param.keys()):
            param = from_params(param, mode=mode)
        else:
            param = {k: _init_param(v, mode) for k, v in param.items()}
    return param


def from_params(params: Dict, mode: str = 'infer', **kwargs) -> Union[Component, FunctionType]:
    """Builds and returns the Component from corresponding dictionary of parameters."""
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
        from deeppavlov.core.commands.infer import build_model
        refs = _refs.copy()
        _refs.clear()
        config = parse_config(expand_path(config_params['config_path']), config_params.get('overwrite'))
        model = build_model(config)
        _refs.clear()
        _refs.update(refs)
        try:
            _refs[config_params['id']] = model
        except KeyError:
            pass
        return model

    cls_name = config_params.pop('class_name', None)
    if not cls_name:
        e = ConfigError('Component config has no `class_name` nor `ref` fields')
        log.exception(e)
        raise e
    obj = get_model(cls_name)

    if inspect.isclass(obj):
        # find the submodels params recursively
        config_params = {k: _init_param(v, mode) for k, v in config_params.items()}
        try:
            spec = inspect.getfullargspec(obj)
            if 'mode' in spec.args + spec.kwonlyargs or spec.varkw is not None:
                kwargs['mode'] = mode

            component = obj(**dict(config_params, **kwargs))
            try:
                _refs[config_params['id']] = component
            except KeyError:
                pass
        except Exception:
            log.exception("Exception in {}".format(obj))
            raise
    else:
        component = obj

    return component
