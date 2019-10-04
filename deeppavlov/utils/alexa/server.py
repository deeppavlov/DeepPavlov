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
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from queue import Queue
from typing import Union, Optional

import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.utils.alexa.bot import AlexaBot
from deeppavlov.utils.alexa.request_parameters import data_body, cert_chain_url_header, signature_header
from deeppavlov.utils.server.server import get_ssl_params, redirect_root_to_docs

SERVER_CONFIG_FILENAME = 'server_config.json'

AMAZON_CERTIFICATE_LIFETIME = timedelta(hours=1)

log = getLogger(__name__)
uvicorn_log = getLogger('uvicorn')
app = FastAPI()


def run_alexa_default_agent(model_config: Union[str, Path, dict],
                            stateful: bool = False,
                            port: Optional[int] = None,
                            https: bool = False,
                            ssl_key: Optional[str] = None,
                            ssl_cert: Optional[str] = None,
                            default_skill_wrap: bool = True) -> None:
    """Initiates FastAPI web service with Alexa skill.

    Allows raise Alexa web service with DeepPavlov config in backend.

    Args:
        model_config: DeepPavlov config path.
        stateful: Stateful mode flag.
        port: FastAPI web service port.
        https: Flag for running Alexa skill service in https mode.
        ssl_key: SSL key file path.
        ssl_cert: SSL certificate file path.
        default_skill_wrap: Wrap with default skill flag.

    """
    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME).resolve()
    server_params = read_json(server_config_path)

    host = server_params['common_defaults']['host']
    port = port or server_params['common_defaults']['port']
    alexa_server_params = server_params['alexa_defaults']

    alexa_server_params['stateful'] = stateful or server_params['common_defaults']['stateful']
    alexa_server_params['amazon_cert_lifetime'] = AMAZON_CERTIFICATE_LIFETIME

    ssl_config = get_ssl_params(server_params['common_defaults'], https, ssl_key=ssl_key, ssl_cert=ssl_cert)

    input_q = Queue()
    output_q = Queue()

    bot = AlexaBot(model_config, default_skill_wrap, alexa_server_params, input_q, output_q)
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

    uvicorn.run(app, host=host, port=port, logger=uvicorn_log, ssl_version=ssl_config.version,
                ssl_keyfile=ssl_config.keyfile, ssl_certfile=ssl_config.certfile)
    bot.join()
