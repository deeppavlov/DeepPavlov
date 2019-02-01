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

import ssl
from collections import namedtuple
from logging import getLogger
from pathlib import Path
from typing import Union, Optional

from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS

from deeppavlov import build_model
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.default_rich_content_processor import DefaultRichContentWrapper
from deeppavlov.core.agent import Agent
from deeppavlov.core.agent.rich_content import RichMessage
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.skills.default_skill.default_skill import DefaultStatelessSkill
from utils.server_utils.server import get_server_params

SERVER_CONFIG_FILENAME = 'server_config.json'

log = getLogger(__name__)

app = Flask(__name__)
Swagger(app)
CORS(app)

DialogID = namedtuple('DialogID', ['user_id', 'session_id'])


def interact_alice(agent: Agent):
    """
    Exchange messages between basic pipelines and the Yandex.Dialogs service.
    If the pipeline returns multiple values, only the first one is forwarded to Yandex.
    """
    data = request.get_json()
    text = data['request'].get('command', '').strip()
    payload = data['request'].get('payload')

    session_id = data['session']['session_id']
    user_id = data['session']['user_id']
    message_id = data['session']['message_id']

    dialog_id = DialogID(user_id, session_id)

    response = {
        'response': {
            'end_session': True,
            'text': ''
        },
        "session": {
            'session_id': session_id,
            'message_id': message_id,
            'user_id': user_id
        },
        'version': '1.0'
    }

    agent_response: Union[str, RichMessage] = agent([payload or text], [dialog_id])[0]
    if isinstance(agent_response, RichMessage):
        response['response']['text'] = '\n'.join([j['content']
                                                  for j in agent_response.json()
                                                  if j['type'] == 'plain_text'])
    else:
        response['response']['text'] = str(agent_response)

    return jsonify(response), 200


def start_alice_server(model_config, https=False, ssl_key=None, ssl_cert=None, port=None):
    server_config_path = get_settings_path() / SERVER_CONFIG_FILENAME
    server_params = get_server_params(server_config_path, model_config)

    https = https or server_params['https']

    if not https:
        ssl_key = ssl_cert = None
    else:
        ssh_key = Path(ssl_key or server_params['https_key_path']).resolve()
        if not ssh_key.is_file():
            e = FileNotFoundError('Ssh key file not found: please provide correct path in --key param or '
                                  'https_key_path param in server configuration file')
            log.error(e)
            raise e

        ssh_cert = Path(ssl_cert or server_params['https_cert_path']).resolve()
        if not ssh_cert.is_file():
            e = FileNotFoundError('Ssh certificate file not found: please provide correct path in --cert param or '
                                  'https_cert_path param in server configuration file')
            log.error(e)
            raise e

    host = server_params['host']
    port = port or server_params['port']
    model_endpoint = server_params['model_endpoint']

    model = build_model(model_config)
    skill = DefaultStatelessSkill(model, lang='ru')
    agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())

    start_agent_server(agent, host, port, model_endpoint, ssl_key, ssl_cert)


def start_agent_server(agent: Agent, host: str, port: int, endpoint: str,
                       ssl_key: Optional[Path] = None,
                       ssl_cert: Optional[Path] = None):
    if ssl_key and ssl_cert:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssh_key_path = Path(ssl_key).resolve()
        ssh_cert_path = Path(ssl_cert).resolve()
        ssl_context.load_cert_chain(ssh_cert_path, ssh_key_path)
    else:
        ssl_context = None

    @app.route('/')
    def index():
        return redirect('/apidocs/')

    endpoint_description = {
        'description': 'A model endpoint',
        'parameters': [
            {
                'name': 'data',
                'in': 'body',
                'required': 'true',
                'example': {
                    'meta': {
                        'locale': 'ru-RU',
                        'timezone': 'Europe/Moscow',
                        "client_id": 'ru.yandex.searchplugin/5.80 (Samsung Galaxy; Android 4.4)'
                    },
                    'request': {
                        'command': 'где ближайшее отделение',
                        'original_utterance': 'Алиса спроси у Сбербанка где ближайшее отделение',
                        'type': 'SimpleUtterance',
                        'markup': {
                            'dangerous_context': True
                        },
                        'payload': {}
                    },
                    'session': {
                        'new': True,
                        'message_id': 4,
                        'session_id': '2eac4854-fce721f3-b845abba-20d60',
                        'skill_id': '3ad36498-f5rd-4079-a14b-788652932056',
                        'user_id': 'AC9WC3DF6FCE052E45A4566A48E6B7193774B84814CE49A922E163B8B29881DC'
                    },
                    'version': '1.0'
                }
            }
        ],
        'responses': {
            "200": {
                "description": "A model response"
            }
        }
    }

    @app.route(endpoint, methods=['POST'])
    @swag_from(endpoint_description)
    def answer():
        return interact_alice(agent)

    app.run(host=host, port=port, threaded=False, ssl_context=ssl_context)
