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
from typing import Union, Optional

import uvicorn
from fastapi import FastAPI

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.utils.connector import MSBot
from deeppavlov.utils.server.server import get_ssl_params, redirect_root_to_docs

SERVER_CONFIG_FILENAME = 'server_config.json'

log = getLogger(__name__)
uvicorn_log = getLogger('uvicorn')
app = FastAPI()


def run_ms_bf_default_agent(model_config: Union[str, Path, dict],
                            app_id: Optional[str],
                            app_secret: Optional[str],
                            port: Optional[int] = None,
                            https: bool = False,
                            ssl_key: Optional[str] = None,
                            ssl_cert: Optional[str] = None) -> None:

    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME).resolve()
    server_params = read_json(server_config_path)

    host = server_params['common_defaults']['host']
    port = port or server_params['common_defaults']['port']

    ms_bf_server_params = server_params['ms_bot_framework_defaults']

    ms_bf_server_params['auth_payload']['client_id'] = app_id or ms_bf_server_params['auth_payload']['client_id']
    if not ms_bf_server_params['auth_payload']['client_id']:
        e = ValueError('Microsoft Bot Framework app id required: initiate -i param '
                       'or auth_payload.client_id param in server configuration file')
        log.error(e)
        raise e

    ms_bf_server_params['auth_payload']['client_secret'] = app_secret or ms_bf_server_params['auth_payload']['client_secret']
    if not ms_bf_server_params['auth_payload']['client_secret']:
        e = ValueError('Microsoft Bot Framework app secret required: initiate -s param '
                       'or auth_payload.client_secret param in server configuration file')
        log.error(e)
        raise e

    ssl_config = get_ssl_params(server_params['common_defaults'], https, ssl_key=ssl_key, ssl_cert=ssl_cert)

    input_q = Queue()
    bot = MSBot(model_config, ms_bf_server_params, input_q)
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
