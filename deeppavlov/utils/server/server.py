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

import logging
from pathlib import Path
from ssl import PROTOCOL_TLSv1_2
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.utils import generate_operation_id_for_path
from pydantic import BaseConfig, BaseModel, Schema
from pydantic.fields import Field
from pydantic.main import MetaModel
from starlette.responses import JSONResponse, RedirectResponse

from deeppavlov.core.agent.dialog_logger import DialogLogger
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.core.data.utils import check_nested_dict_keys, jsonify_data

SERVER_CONFIG_FILENAME = 'server_config.json'


class ProbeFilter(logging.Filter):
    """ProbeFilter class is used to filter POST requests to /probe endpoint from logs."""
    def filter(self, record: logging.LogRecord) -> bool:
        """To log the record method should return True."""
        return 'POST /probe HTTP' not in record.getMessage()


log = logging.getLogger(__name__)
uvicorn_log = logging.getLogger('uvicorn')
uvicorn_log.addFilter(ProbeFilter())
app = FastAPI(__file__)

dialog_logger = DialogLogger(agent_name='dp_api')


def get_server_params(server_config_path: Path, model_config: Path) -> Dict:
    server_config = read_json(server_config_path)
    model_config = parse_config(model_config)

    server_params = server_config['common_defaults']

    if check_nested_dict_keys(model_config, ['metadata', 'labels', 'server_utils']):
        model_tag = model_config['metadata']['labels']['server_utils']
        if model_tag in server_config['model_defaults']:
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


def interact(model: Chainer, payload: Dict[str, Optional[List]]) -> JSONResponse:
    model_args = payload.values()
    dialog_logger.log_in(payload)
    error_msg = None
    lengths = {len(model_arg) for model_arg in model_args if model_arg is not None}

    if not lengths:
        error_msg = 'got empty request'
    elif 0 in lengths:
        error_msg = 'dot empty array as model argument'
    elif len(lengths) > 1:
        error_msg = 'got several different batch sizes'

    if error_msg is not None:
        log.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    batch_size = next(iter(lengths))
    model_args = [arg or [None] * batch_size for arg in model_args]

    prediction = model(*model_args)
    if len(model.out_params) == 1:
        prediction = [prediction]
    prediction = list(zip(*prediction))
    result = jsonify_data(prediction)
    dialog_logger.log_out(result)
    return JSONResponse(result)


def test_interact(model: Chainer, payload: Dict[str, Optional[List]]) -> JSONResponse:
    model_args = [arg or ["Test string."] for arg in payload.values()]
    try:
        _ = model(*model_args)
        return JSONResponse(["Test passed"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=e)


def start_model_server(model_config: Path, https: bool = False, ssl_key: Optional[str] = None,
                       ssl_cert: Optional[str] = None, port: Optional[int] = None) -> None:
    server_config_path = get_settings_path() / SERVER_CONFIG_FILENAME
    server_params = get_server_params(server_config_path, model_config)

    host = server_params['host']
    port = port or server_params['port']
    model_endpoint = server_params['model_endpoint']
    docs_endpoint = server_params['docs_endpoint']
    model_args_names = server_params['model_args_names']

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

        ssl_version = PROTOCOL_TLSv1_2
        ssl_keyfile = str(ssh_key_path)
        ssl_certfile = str(ssh_cert_path)
    else:
        ssl_version = None
        ssl_keyfile = None
        ssl_certfile = None

    model = build_model(model_config)

    def batch_decorator(cls: MetaModel) -> MetaModel:
        cls.__annotations__ = {arg_name: List[str] for arg_name in model_args_names}
        cls.__fields__ = {arg_name: Field(name=arg_name, type_=List[str], class_validators=None,
                                          model_config=BaseConfig, required=False, schema=Schema(None))
                          for arg_name in model_args_names}
        return cls

    @batch_decorator
    class Batch(BaseModel):
        pass

    @app.get('/', include_in_schema=False)
    async def redirect_to_docs() -> RedirectResponse:
        operation_id = generate_operation_id_for_path(name='answer', path=model_endpoint, method='post')
        response = RedirectResponse(url=f'{docs_endpoint}#/default/{operation_id}')
        return response

    @app.post(model_endpoint, status_code=200, summary='A model endpoint')
    async def answer(item: Batch) -> JSONResponse:
        return interact(model, item.dict())

    @app.post('/probe', status_code=200, include_in_schema=False)
    async def probe(item: Batch) -> JSONResponse:
        return test_interact(model, item.dict())

    @app.get('/api', status_code=200, summary='Model argument names')
    async def api() -> JSONResponse:
        return JSONResponse(model_args_names)

    uvicorn.run(app, host=host, port=port, logger=uvicorn_log,
                ssl_version=ssl_version, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
