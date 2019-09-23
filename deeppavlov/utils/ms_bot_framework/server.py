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
from starlette.responses import JSONResponse

from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.default_rich_content_processor import DefaultRichContentWrapper
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.skills.default_skill.default_skill import DefaultStatelessSkill
from deeppavlov.utils.ms_bot_framework.bot import Bot
from deeppavlov.utils.server.server import get_ssl_params, redirect_root_do_docs

SERVER_CONFIG_FILENAME = 'server_config.json'

AUTH_URL = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
AUTH_HOST = "login.microsoftonline.com"
AUTH_CONTENT_TYPE = "application/x-www-form-urlencoded"
AUTH_GRANT_TYPE = "client_credentials"
AUTH_SCOPE = "https://api.botframework.com/.default"

log = getLogger(__name__)
uvicorn_log = getLogger('uvicorn')
app = FastAPI()


def run_ms_bf_default_agent(model_config: Union[str, Path, dict],
                            app_id: str,
                            app_secret: str,
                            multi_instance: bool = False,
                            stateful: bool = False,
                            port: Optional[int] = None,
                            https: bool = False,
                            ssl_key: Optional[str] = None,
                            ssl_cert: Optional[str] = None,
                            default_skill_wrap: bool = True) -> None:

    def get_default_agent() -> DefaultAgent:
        model = build_model(model_config)
        skill = DefaultStatelessSkill(model) if default_skill_wrap else model
        agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())
        return agent

    run_ms_bot_framework_server(agent_generator=get_default_agent,
                                app_id=app_id,
                                app_secret=app_secret,
                                multi_instance=multi_instance,
                                stateful=stateful,
                                port=port,
                                https=https,
                                ssl_key=ssl_key,
                                ssl_cert=ssl_cert)


def run_ms_bot_framework_server(agent_generator: callable,
                                app_id: str,
                                app_secret: str,
                                multi_instance: bool = False,
                                stateful: bool = False,
                                port: Optional[int] = None,
                                https: bool = False,
                                ssl_key: Optional[str] = None,
                                ssl_cert: Optional[str] = None) -> None:

    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME).resolve()
    server_params = read_json(server_config_path)

    host = server_params['common_defaults']['host']
    port = port or server_params['common_defaults']['port']

    ms_bf_server_params = server_params['ms_bot_framework_defaults']

    ms_bf_server_params['multi_instance'] = multi_instance or server_params['common_defaults']['multi_instance']
    ms_bf_server_params['stateful'] = stateful or server_params['common_defaults']['stateful']

    ms_bf_server_params['auth_url'] = AUTH_URL
    ms_bf_server_params['auth_host'] = AUTH_HOST
    ms_bf_server_params['auth_content_type'] = AUTH_CONTENT_TYPE
    ms_bf_server_params['auth_grant_type'] = AUTH_GRANT_TYPE
    ms_bf_server_params['auth_scope'] = AUTH_SCOPE

    ms_bf_server_params['auth_app_id'] = app_id or ms_bf_server_params['auth_app_id']
    if not ms_bf_server_params['auth_app_id']:
        e = ValueError('Microsoft Bot Framework app id required: initiate -i param '
                       'or auth_app_id param in server configuration file')
        log.error(e)
        raise e

    ms_bf_server_params['auth_app_secret'] = app_secret or ms_bf_server_params['auth_app_secret']
    if not ms_bf_server_params['auth_app_secret']:
        e = ValueError('Microsoft Bot Framework app secret required: initiate -s param '
                       'or auth_app_secret param in server configuration file')
        log.error(e)
        raise e

    ssl_config = get_ssl_params(server_params, https, ssl_key=ssl_key, ssl_cert=ssl_cert)

    input_q = Queue()
    bot = Bot(agent_generator, ms_bf_server_params, input_q)
    bot.start()

    endpoint = '/v3/conversations'
    redirect_root_do_docs(app, 'answer', endpoint, 'post')

    @app.post(endpoint)
    async def answer(activity: dict) -> JSONResponse:
        bot.input_queue.put(activity)
        return JSONResponse({})

    uvicorn.run(app, host=host, port=port, logger=uvicorn_log, ssl_version=ssl_config.version,
                ssl_keyfile=ssl_config.keyfile, ssl_certfile=ssl_config.certfile)
