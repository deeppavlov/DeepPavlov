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

from deeppavlov.utils.connector import MSBot, get_connector_params
from deeppavlov.utils.server.server import get_ssl_params, redirect_root_to_docs

log = getLogger(__name__)
uvicorn_log = getLogger('uvicorn')
app = FastAPI()


def run_ms_bf_default_agent(model_config: Path,
                            app_id: Optional[str],
                            app_secret: Optional[str],
                            port: Optional[int] = None,
                            https: bool = False,
                            ssl_key: Optional[str] = None,
                            ssl_cert: Optional[str] = None) -> None:

    connector_params = get_connector_params('ms_bot_framework', model_config)

    host = connector_params['host']
    port = port or connector_params['port']

    auth_params = {
        "auth_headers": {
          "Host": "login.microsoftonline.com",
          "Content-Type": "application/x-www-form-urlencoded"
        },
        "auth_payload": {
          "grant_type": "client_credentials",
          "scope": "https://api.botframework.com/.default",
          "client_id": app_id or connector_params['client_id'],
          "client_secret": app_secret or connector_params['client_secret']
        }
    }
    connector_params.update(auth_params)

    if not connector_params['auth_payload']['client_id']:
        e = ValueError('Microsoft Bot Framework app id required: initiate -i param '
                       'or auth_payload.client_id param in server configuration file')
        log.error(e)
        raise e

    if not connector_params['auth_payload']['client_secret']:
        e = ValueError('Microsoft Bot Framework app secret required: initiate -s param '
                       'or auth_payload.client_secret param in server configuration file')
        log.error(e)
        raise e

    ssl_config = get_ssl_params(connector_params, https, ssl_key=ssl_key, ssl_cert=ssl_cert)

    input_q = Queue()
    bot = MSBot(model_config, connector_params, input_q)
    bot.start()

    endpoint = '/v3/conversations'
    redirect_root_to_docs(app, 'answer', endpoint, 'post')

    @app.post(endpoint)
    async def answer(activity: dict) -> dict:
        bot.input_queue.put(activity)
        return {}

    uvicorn.run(app, host=host, port=port, logger=uvicorn_log, ssl_version=ssl_config.version,
                ssl_keyfile=ssl_config.keyfile, ssl_certfile=ssl_config.certfile)
    bot.join()
