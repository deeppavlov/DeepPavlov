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
from datetime import timedelta
from pathlib import Path
from queue import Queue
from typing import Union, Optional

from flask import Flask, request, jsonify, redirect
from flasgger import Swagger, swag_from
from flask_cors import CORS

from utils.alexa.bot import Bot
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.default_rich_content_processor import DefaultRichContentWrapper
from deeppavlov.skills.default_skill.default_skill import DefaultStatelessSkill

SERVER_CONFIG_FILENAME = 'server_config.json'

AMAZON_CERTIFICATE_LIFETIME = timedelta(hours=1)

log = get_logger(__name__)

app = Flask(__name__)
Swagger(app)
CORS(app)


def run_alexa_default_agent(model_config: Union[str, Path, dict], multi_instance: bool = False,
                            stateful: bool = False, port: Optional[int] = None, https: bool = False,
                            ssl_key: str = None, ssl_cert: str = None) -> None:
    """Creates Alexa agents factory and initiates Alexa web service.

    Wrapper around run_alexa_server. Allows raise Alexa web service with
    DeepPavlov config in backend.

    Args:
        model_config: DeepPavlov config path.
        multi_instance: Multi instance mode flag.
        stateful: Stateful mode flag.
        port: Flask web service port.
        https: Flag for running Alexa skill service in https mode.
        ssl_key: SSL key file path.
        ssl_cert: SSL certificate file path.
    """
    def get_default_agent() -> DefaultAgent:
        model = build_model(model_config)
        skill = DefaultStatelessSkill(model)
        agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())
        return agent

    run_alexa_server(agent_generator=get_default_agent,
                     multi_instance=multi_instance,
                     stateful=stateful,
                     port=port,
                     https=https,
                     ssl_key=ssl_key,
                     ssl_cert=ssl_cert)


def run_alexa_server(agent_generator: callable, multi_instance: bool = False,
                     stateful: bool = False, port: Optional[int] = None, https: bool = False,
                     ssl_key: str = None, ssl_cert: str = None) -> None:
    """Initiates Flask web service with Alexa skill.

    Args:
        agent_generator: Callback Alexa agents factory.
        multi_instance: Multi instance mode flag.
        stateful: Stateful mode flag.
        port: Flask web service port.
        https: Flag for running Alexa skill service in https mode.
        ssl_key: SSL key file path.
        ssl_cert: SSL certificate file path.
    """
    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME).resolve()
    server_params = read_json(server_config_path)

    host = server_params['common_defaults']['host']
    port = port or server_params['common_defaults']['port']

    alexa_server_params = server_params['alexa_defaults']

    alexa_server_params['multi_instance'] = multi_instance or server_params['common_defaults']['multi_instance']
    alexa_server_params['stateful'] = stateful or server_params['common_defaults']['stateful']
    alexa_server_params['amazon_cert_lifetime'] = AMAZON_CERTIFICATE_LIFETIME

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

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssl_context.load_cert_chain(ssh_cert_path, ssh_key_path)
    else:
        ssl_context = None

    input_q = Queue()
    output_q = Queue()

    bot = Bot(agent_generator, alexa_server_params, input_q, output_q)
    bot.start()

    endpoint_description = {
        'description': 'Amazon Alexa custom service endpoint',
        'parameters': [
            {
                'name': 'data',
                'in': 'body',
                'required': 'true'
            }
        ],
        'responses': {
            "200": {
                "description": "A model response"
            }
        }
    }

    @app.route('/')
    def index():
        return redirect('/apidocs/')

    @app.route('/interact', methods=['POST'])
    @swag_from(endpoint_description)
    def handle_request():
        request_body: bytes = request.get_data()
        signature_chain_url: str = request.headers.get('Signaturecertchainurl')
        signature: str = request.headers.get('Signature')
        alexa_request: dict = request.get_json()

        request_dict = {
            'request_body': request_body,
            'signature_chain_url': signature_chain_url,
            'signature': signature,
            'alexa_request': alexa_request
        }

        bot.input_queue.put(request_dict)
        response: dict = bot.output_queue.get()
        response_code = 400 if 'error' in response.keys() else 200

        return jsonify(response), response_code

    app.run(host=host, port=port, threaded=True, ssl_context=ssl_context)
