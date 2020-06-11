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

from logging import getLogger
from pathlib import Path
from queue import Queue
from typing import Optional

import uvicorn
from fastapi import FastAPI

from deeppavlov.core.common.log import log_config
from deeppavlov.utils.connector import MSBot
from deeppavlov.utils.server import get_server_params, get_ssl_params, redirect_root_to_docs

log = getLogger(__name__)
app = FastAPI()


def start_ms_bf_server(model_config: Path,
                       app_id: Optional[str],
                       app_secret: Optional[str],
                       port: Optional[int] = None,
                       https: Optional[bool] = None,
                       ssl_key: Optional[str] = None,
                       ssl_cert: Optional[str] = None) -> None:

    server_params = get_server_params(model_config)

    host = server_params['host']
    port = port or server_params['port']

    ssl_config = get_ssl_params(server_params, https, ssl_key=ssl_key, ssl_cert=ssl_cert)

    input_q = Queue()
    bot = MSBot(model_config, input_q, app_id, app_secret)
    bot.start()

    endpoint = '/v3/conversations'
    redirect_root_to_docs(app, 'answer', endpoint, 'post')

    @app.post(endpoint)
    async def answer(activity: dict) -> dict:
        bot.input_queue.put(activity)
        return {}

    uvicorn.run(app, host=host, port=port, log_config=log_config, ssl_version=ssl_config.version,
                ssl_keyfile=ssl_config.keyfile, ssl_certfile=ssl_config.certfile)
    bot.join()
