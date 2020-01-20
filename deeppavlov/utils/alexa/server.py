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
import json
from logging import getLogger
from pathlib import Path
from queue import Queue
from typing import Union, Optional

import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse

from deeppavlov.core.common.log import log_config
from deeppavlov.utils.alexa.request_parameters import data_body, cert_chain_url_header, signature_header
from deeppavlov.utils.connector import AlexaBot
from deeppavlov.utils.server import get_ssl_params, redirect_root_to_docs, get_server_params

log = getLogger(__name__)
app = FastAPI()


def start_alexa_server(model_config: Union[str, Path, dict],
                       port: Optional[int] = None,
                       https: Optional[bool] = None,
                       ssl_key: Optional[str] = None,
                       ssl_cert: Optional[str] = None) -> None:
    """Initiates FastAPI web service with Alexa skill.

    Allows raise Alexa web service with DeepPavlov config in backend.

    Args:
        model_config: DeepPavlov config path.
        port: FastAPI web service port.
        https: Flag for running Alexa skill service in https mode.
        ssl_key: SSL key file path.
        ssl_cert: SSL certificate file path.

    """
    server_params = get_server_params(model_config)

    host = server_params['host']
    port = port or server_params['port']

    ssl_config = get_ssl_params(server_params, https, ssl_key=ssl_key, ssl_cert=ssl_cert)

    input_q = Queue()
    output_q = Queue()

    bot = AlexaBot(model_config, input_q, output_q)
    bot.start()

    endpoint = '/interact'
    redirect_root_to_docs(app, 'interact', endpoint, 'post')

    @app.post(endpoint, summary='Amazon Alexa custom service endpoint', response_description='A model response')
    async def interact(data: dict = data_body,
                       signature: str = signature_header,
                       signature_chain_url: str = cert_chain_url_header) -> JSONResponse:
        # It is necessary for correct data validation to serialize data to a JSON formatted string with separators.
        request_dict = {
            'request_body': json.dumps(data, separators=(',', ':')).encode('utf-8'),
            'signature_chain_url': signature_chain_url,
            'signature': signature,
            'alexa_request': data
        }

        bot.input_queue.put(request_dict)
        loop = asyncio.get_event_loop()
        response: dict = await loop.run_in_executor(None, bot.output_queue.get)
        response_code = 400 if 'error' in response.keys() else 200
        return JSONResponse(response, status_code=response_code)

    uvicorn.run(app, host=host, port=port, log_config=log_config, ssl_version=ssl_config.version,
                ssl_keyfile=ssl_config.keyfile, ssl_certfile=ssl_config.certfile)
    bot.join()
