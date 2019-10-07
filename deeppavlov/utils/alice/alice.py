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
from logging import getLogger
from pathlib import Path
from queue import Queue
from typing import Optional

import uvicorn
from fastapi import FastAPI

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.utils.alice.request_parameters import data_body
from deeppavlov.utils.connector import AliceBot
from deeppavlov.utils.server.server import SSLConfig, get_server_params, get_ssl_params, redirect_root_to_docs

SERVER_CONFIG_FILENAME = 'server_config.json'

log = getLogger(__name__)
uvicorn_log = getLogger('uvicorn')
app = FastAPI()


def start_alice_server(model_config: Path,
                       https: bool = False,
                       ssl_key: Optional[str] = None,
                       ssl_cert: Optional[str] = None,
                       port: Optional[int] = None) -> None:
    server_config_path = get_settings_path() / SERVER_CONFIG_FILENAME
    server_params = get_server_params(server_config_path, model_config)
    alice_server_params = read_json(server_config_path)['alice_defaults']

    https = https or server_params['https']
    host = server_params['host']
    port = port or server_params['port']
    model_endpoint = server_params['model_endpoint']

    ssl_config = get_ssl_params(server_params, https, ssl_key=ssl_key, ssl_cert=ssl_cert)

    input_q = Queue()
    output_q = Queue()

    bot = AliceBot(model_config, alice_server_params, input_q, output_q)

    start_agent_server(bot, host, port, model_endpoint, ssl_config=ssl_config)


def start_agent_server(bot: AliceBot,
                       host: str,
                       port: int,
                       endpoint: str,
                       ssl_key: Optional[str] = None,
                       ssl_cert: Optional[str] = None,
                       ssl_config: Optional[SSLConfig] = None) -> None:

    if ssl_key and ssl_cert and ssl_config:
        raise ValueError('ssl_key, ssl_cert, ssl_config was assigned at the same time. Please, use either'
                         'ssl_config or ssl_key and ssl_cert')

    if ssl_key and ssl_cert:
        ssl_config = get_ssl_params({}, True, ssl_key=ssl_key, ssl_cert=ssl_cert)
    else:
        ssl_config = ssl_config or get_ssl_params({}, False, ssl_key=None, ssl_cert=None)

    bot.start()

    redirect_root_to_docs(app, 'answer', endpoint, 'post')

    @app.post(endpoint, summary='A model endpoint', response_description='A model response')
    async def answer(data: dict = data_body) -> dict:
        loop = asyncio.get_event_loop()
        bot.input_queue.put(data)
        response: dict = await loop.run_in_executor(None, bot.output_queue.get)
        return response

    uvicorn.run(app, host=host, port=port, logger=uvicorn_log, ssl_version=ssl_config.version,
                ssl_keyfile=ssl_config.keyfile, ssl_certfile=ssl_config.certfile)
    bot.join()
