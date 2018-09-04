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

import importlib
import inspect
from typing import Dict

from deeppavlov.core.commands.utils import expand_path, get_deeppavlov_root, set_deeppavlov_root
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import get_model, cls_from_str
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


def _init_param(param, mode):
    if isinstance(param, str):
        param = _resolve(param)
    elif isinstance(param, (list, tuple)):
        param = [_init_param(p, mode) for p in param]
    elif isinstance(param, dict):
        if {'ref', 'name', 'class', 'config_path'}.intersection(param.keys()):
            param = from_params(param, mode=mode)
        else:
            param = {k: _init_param(v, mode) for k, v in param.items()}
    return param


def from_params(params: Dict, mode: str = 'infer', **kwargs) -> Component:
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
        from deeppavlov.core.commands.infer import build_model_from_config
        deeppavlov_root = get_deeppavlov_root()
        refs = _refs.copy()
        _refs.clear()
        config = read_json(expand_path(config_params['config_path']))
        model = build_model_from_config(config, as_component=True)
        set_deeppavlov_root({'deeppavlov_root': deeppavlov_root})
        _refs.clear()
        _refs.update(refs)
        return model

    elif 'class' in config_params:
        cls = cls_from_str(config_params.pop('class'))
    else:
        cls_name = config_params.pop('name', None)
        if not cls_name:
            e = ConfigError('Component config has no `name` nor `ref` or `class` fields')
            log.exception(e)
            raise e
        cls = get_model(cls_name)

    # find the submodels params recursively
    config_params = {k: _init_param(v, mode) for k, v in config_params.items()}

    try:
        spec = inspect.getfullargspec(cls)
        if 'mode' in spec.args+spec.kwonlyargs or spec.varkw is not None:
            kwargs['mode'] = mode

        component = cls(**dict(config_params, **kwargs))
        try:
            _refs[config_params['id']] = component
        except KeyError:
            pass
    except Exception:
        log.exception("Exception in {}".format(cls))
        raise

    return component
