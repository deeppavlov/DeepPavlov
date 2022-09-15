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

import asyncio
import os
from collections import namedtuple
from logging import getLogger
from pathlib import Path
from ssl import PROTOCOL_TLSv1_2
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.utils import generate_operation_id_for_path
from pydantic import BaseConfig, BaseModel
from pydantic.fields import Field, ModelField
from pydantic.main import ModelMetaclass
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import log_config
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.core.data.utils import check_nested_dict_keys, jsonify_data
from deeppavlov.utils.connector import DialogLogger
from deeppavlov.utils.server.metrics import metrics, PrometheusMiddleware

SERVER_CONFIG_PATH = get_settings_path() / 'server_config.json'
SSLConfig = namedtuple('SSLConfig', ['version', 'keyfile', 'certfile'])

log = getLogger(__name__)
dialog_logger = DialogLogger(logger_name='rest_api')

COMPATIBILITY_MODE = os.getenv('COMPATIBILITY_MODE', False)

if COMPATIBILITY_MODE is not False:
    log.warning('DeepPavlov riseapi mode will use the old model response data format used up and including 1.0.0rc1.\n'
                'COMPATIBILITY_MODE will be removed in the DeepPavlov 1.2.0.\n'
                'Please, update your client code according to the new format.')

app = FastAPI()

app.add_middleware(
    PrometheusMiddleware,
    ignore_paths=('/', '/metrics', '/api', '/probe', '/docs', '/openapi.json')
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

app.add_route("/metrics", metrics)


def get_server_params(model_config: Union[str, Path]) -> Dict:
    server_config = read_json(SERVER_CONFIG_PATH)
    model_config = parse_config(model_config)

    server_params = server_config['common_defaults']

    if check_nested_dict_keys(model_config, ['metadata', 'server_utils']):
        model_tag = model_config['metadata']['server_utils']
        if check_nested_dict_keys(server_config, ['model_defaults', model_tag]):
            model_defaults = server_config['model_defaults'][model_tag]
            for param_name in model_defaults.keys():
                if model_defaults[param_name]:
                    server_params[param_name] = model_defaults[param_name]

    server_params['model_endpoint'] = server_params.get('model_endpoint', '/model')

    arg_names = server_params['model_args_names'] or model_config['chainer']['in']
    if isinstance(arg_names, str):
        arg_names = [arg_names]
    server_params['model_args_names'] = arg_names

    return server_params


def get_ssl_params(server_params: dict,
                   https: Optional[bool],
                   ssl_key: Optional[str],
                   ssl_cert: Optional[str]) -> SSLConfig:
    https = https or server_params['https']
    if https:
        ssh_key_path = Path(ssl_key or server_params['https_key_path']).resolve()
        if not ssh_key_path.is_file():
            e = FileNotFoundError('Ssh key file not found: please provide correct path in --key param or '
                                  'https_key_path param in server configuration file')
            log.error(e)
            raise e

        ssh_cert_path = Path(ssl_cert or server_params['https_cert_path']).resolve()
        if not ssh_cert_path.is_file():
            e = FileNotFoundError('Ssh certificate file not found: please provide correct path in --cert param or '
                                  'https_cert_path param in server configuration file')
            log.error(e)
            raise e

        ssl_config = SSLConfig(version=PROTOCOL_TLSv1_2, keyfile=str(ssh_key_path), certfile=str(ssh_cert_path))
    else:
        ssl_config = SSLConfig(None, None, None)

    return ssl_config


def redirect_root_to_docs(fast_app: FastAPI, func_name: str, endpoint: str, method: str) -> None:
    """Adds api route to server that redirects user from root to docs with opened `endpoint` description."""

    @fast_app.get('/', include_in_schema=False)
    async def redirect_to_docs() -> RedirectResponse:
        operation_id = generate_operation_id_for_path(name=func_name, path=endpoint, method=method)
        response = RedirectResponse(url=f'/docs#/default/{operation_id}')
        return response


def interact(model: Chainer, payload: Dict[str, Optional[List]]) -> List:
    model_args = payload.values()
    dialog_logger.log_in(payload)
    error_msg = None
    lengths = {len(model_arg) for model_arg in model_args if model_arg is not None}

    if not lengths:
        error_msg = 'got empty request'
    elif 0 in lengths:
        error_msg = 'got empty array as model argument'
    elif len(lengths) > 1:
        error_msg = 'got several different batch sizes'

    if error_msg is not None:
        log.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    batch_size = next(iter(lengths))
    model_args = [arg or [None] * batch_size for arg in model_args]

    prediction = model(*model_args)

    # TODO: remove in 1.2.0
    if COMPATIBILITY_MODE is not False:
        if len(model.out_params) == 1:
            prediction = [prediction]
        prediction = list(zip(*prediction))

    result = jsonify_data(prediction)
    dialog_logger.log_out(result)
    return result


def test_interact(model: Chainer, payload: Dict[str, Optional[List]]) -> List[str]:
    model_args = [arg or ["Test string."] for arg in payload.values()]
    try:
        _ = model(*model_args)
        return ["Test passed"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=repr(e))


def start_model_server(model_config: Path,
                       https: Optional[bool] = None,
                       ssl_key: Optional[str] = None,
                       ssl_cert: Optional[str] = None,
                       port: Optional[int] = None) -> None:

    server_params = get_server_params(model_config)

    host = server_params['host']
    port = port or server_params['port']
    model_endpoint = server_params['model_endpoint']
    model_args_names = server_params['model_args_names']

    ssl_config = get_ssl_params(server_params, https, ssl_key=ssl_key, ssl_cert=ssl_cert)

    model = build_model(model_config)

    def batch_decorator(cls: ModelMetaclass) -> ModelMetaclass:
        cls.__annotations__ = {arg_name: list for arg_name in model_args_names}
        cls.__fields__ = {arg_name: ModelField(name=arg_name, type_=list, class_validators=None,
                                               model_config=BaseConfig, required=False, field_info=Field(None))
                          for arg_name in model_args_names}
        return cls

    @batch_decorator
    class Batch(BaseModel):
        pass

    redirect_root_to_docs(app, 'answer', model_endpoint, 'post')

    model_endpoint_post_example = {arg_name: ['string'] for arg_name in model_args_names}

    @app.post(model_endpoint, summary='A model endpoint')
    async def answer(item: Batch = Body(..., example=model_endpoint_post_example)) -> List:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, interact, model, item.dict())

    @app.post('/probe', include_in_schema=False)
    async def probe(item: Batch) -> List[str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, test_interact, model, item.dict())

    @app.get('/api', summary='Model argument names')
    async def api() -> Dict[str, List[str]]:
        if COMPATIBILITY_MODE is not False:
            return model_args_names
        return {
            'in': model.in_x,
            'out': model.out_params
        }

    uvicorn.run(app, host=host, port=port, log_config=log_config, ssl_version=ssl_config.version,
                ssl_keyfile=ssl_config.keyfile, ssl_certfile=ssl_config.certfile, timeout_keep_alive=20)
