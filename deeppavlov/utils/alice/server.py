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
from typing import Optional, Union

import uvicorn
from fastapi import FastAPI

from deeppavlov.core.common.log import log_config
from deeppavlov.utils.alice.request_parameters import data_body
from deeppavlov.utils.connector import AliceBot
from deeppavlov.utils.server import get_server_params, get_ssl_params, redirect_root_to_docs

log = getLogger(__name__)
app = FastAPI()


def start_alice_server(model_config: Union[str, Path],
                       host: Optional[str] = None,
                       port: Optional[int] = None,
                       endpoint: Optional[str] = None,
                       https: Optional[bool] = None,
                       ssl_key: Optional[str] = None,
                       ssl_cert: Optional[str] = None) -> None:
    server_params = get_server_params(model_config)

    host = host or server_params['host']
    port = port or server_params['port']
    endpoint = endpoint or server_params['model_endpoint']

    ssl_config = get_ssl_params(server_params, https, ssl_key=ssl_key, ssl_cert=ssl_cert)

    input_q = Queue()
    output_q = Queue()

    bot = AliceBot(model_config, input_q, output_q)
    bot.start()

    redirect_root_to_docs(app, 'answer', endpoint, 'post')

    @app.post(endpoint, summary='A model endpoint', response_description='A model response')
    async def answer(data: dict = data_body) -> dict:
        loop = asyncio.get_event_loop()
        bot.input_queue.put(data)
        response: dict = await loop.run_in_executor(None, bot.output_queue.get)
        return response

    uvicorn.run(app, host=host, port=port, log_config=log_config, ssl_version=ssl_config.version,
                ssl_keyfile=ssl_config.keyfile, ssl_certfile=ssl_config.certfile)
    bot.join()
