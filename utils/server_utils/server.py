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
from itertools import islice
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union, Optional

from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify, redirect, Response
from flask_cors import CORS

from deeppavlov.core.agent.dialog_logger import DialogLogger
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.core.data.utils import check_nested_dict_keys, jsonify_data

SERVER_CONFIG_FILENAME = 'server_config.json'

log = getLogger(__name__)

app = Flask(__name__)
Swagger(app)
CORS(app)

dialog_logger = DialogLogger(agent_name='dp_api')


def get_server_params(server_config_path, model_config):
    server_config = read_json(server_config_path)
    model_config = parse_config(model_config)

    server_params = server_config['common_defaults']

    if check_nested_dict_keys(model_config, ['metadata', 'labels', 'server_utils']):
        model_tag = model_config['metadata']['labels']['server_utils']
        if model_tag in server_config['model_defaults']:
            model_defaults = server_config['model_defaults'][model_tag]
            for param_name in model_defaults.keys():
                if model_defaults[param_name]:
                    server_params[param_name] = model_defaults[param_name]

    return server_params


def interact_skill(model: Chainer, batch_size: Optional[int] = None):
    if not request.is_json:
        log.error("request Content-Type header is not application/json")
        return jsonify({
            "error": "request Content-Type header is not application/json"
        }), 400

    data = request.get_json()
    try:
        dialog_states = iter(data['dialogs'])
    except (KeyError, TypeError):
        return jsonify({
            'error': 'illegal payload format'
        }), 500

    responses = []
    while True:
        batch = list(islice(dialog_states, batch_size))
        if not batch:
            break
        try:
            result = model(batch)
        except Exception as e:
            log.error(f'Got an exception when trying to infer the model: {type(e).__name__}: {e}')
            return jsonify({
                'error': f'{type(e).__name__}: {e}'
            }), 500
        if len(model.out_params) == 1:
            result = [result]
        responses += [dict(zip(model.out_params, response)) for response in zip(*result)]

    return jsonify({
        'responses': responses
    }), 200


def interact(model: Chainer, params_names: List[str]) -> Tuple[Response, int]:
    if not request.is_json:
        log.error("request Content-Type header is not application/json")
        return jsonify({
            "error": "request Content-Type header is not application/json"
        }), 400

    model_args = []

    data = request.get_json()
    dialog_logger.log_in(data)
    for param_name in params_names:
        param_value = data.get(param_name)
        if param_value is None or (isinstance(param_value, list) and len(param_value) > 0):
            model_args.append(param_value)
        else:
            log.error(f"nonempty array expected but got '{param_name}'={repr(param_value)}")
            return jsonify({'error': f"nonempty array expected but got '{param_name}'={repr(param_value)}"}), 400

    lengths = {len(i) for i in model_args if i is not None}

    if not lengths:
        log.error('got empty request')
        return jsonify({'error': 'got empty request'}), 400
    elif len(lengths) > 1:
        log.error('got several different batch sizes')
        return jsonify({'error': 'got several different batch sizes'}), 400

    batch_size = list(lengths)[0]
    model_args = [arg or [None] * batch_size for arg in model_args]

    # in case when some parameters were not described in model_args
    model_args += [[None] * batch_size for _ in range(len(model.in_x) - len(model_args))]

    prediction = model(*model_args)
    if len(model.out_params) == 1:
        prediction = [prediction]
    prediction = list(zip(*prediction))
    result = jsonify_data(prediction)
    dialog_logger.log_out(result)
    return jsonify(result), 200


def _get_ssl_context(ssl_key, ssl_cert):
    ssh_key_path = Path(ssl_key).resolve()
    if not ssh_key_path.is_file():
        e = FileNotFoundError('Ssh key file not found: please provide correct path in --key param or '
                              'https_key_path param in server configuration file')
        log.error(e)
        raise e

    ssh_cert_path = Path(ssl_cert).resolve()
    if not ssh_cert_path.is_file():
        e = FileNotFoundError('Ssh certificate file not found: please provide correct path in --cert param or '
                              'https_cert_path param in server configuration file')
        log.error(e)
        raise e

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    ssl_context.load_cert_chain(ssh_cert_path, ssh_key_path)
    return ssl_context


def start_model_server(model_config, https=False, ssl_key=None, ssl_cert=None, *,
                       host=None, port=None, endpoint=None):
    server_config_path = get_settings_path() / SERVER_CONFIG_FILENAME
    server_params = get_server_params(server_config_path, model_config)

    host = host or server_params['host']
    port = port or server_params['port']
    model_endpoint = endpoint or server_params['model_endpoint']
    model_args_names = server_params['model_args_names']

    https = https or server_params['https']

    if https:
        ssl_context = _get_ssl_context(ssl_key or server_params['https_key_path'],
                                       ssl_cert or server_params['https_cert_path'])
    else:
        ssl_context = None

    model = build_model(model_config)

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
                'example': {arg: ['value'] for arg in model_args_names}
            }
        ],
        'responses': {
            "200": {
                "description": "A model response"
            }
        }
    }

    @app.route(model_endpoint, methods=['POST'])
    @swag_from(endpoint_description)
    def answer():
        return interact(model, model_args_names)

    app.run(host=host, port=port, threaded=False, ssl_context=ssl_context)


def skill_server(config: Union[dict, str, Path], https=False, ssl_key=None, ssl_cert=None, *,
                 host: Optional[str] = None, port: Optional[int] = None, endpoint: Optional[str] = None,
                 download: bool = False, batch_size: Optional[int] = None):
    host = host or '0.0.0.0'
    port = port or 80
    endpoint = endpoint or '/skill'
    if batch_size is not None and batch_size < 1:
        log.warning(f'batch_size of {batch_size} is less than 1 and is interpreted as unlimited')
        batch_size = None

    ssl_context = _get_ssl_context(ssl_key, ssl_cert) if https else None

    model = build_model(config, download=download)

    endpoint_description = {
        'description': 'A skill endpoint',
        'parameters': [
            {
                'name': 'data',
                'in': 'body',
                'required': 'true',
                'example': {
                    'version': 0.9,
                    'dialogs': [
                        {
                            'id': '5c65706b0110b377e17eba41',
                            'location': None,
                            'utterances': [
                                {
                                    "id": "5c62f7330110b36bdd1dc5d7",
                                    "text": "Привет!",
                                    "user_id": "5c62f7330110b36bdd1dc5d5",
                                    "annotations": {
                                        "ner": [
                                        ],
                                        "coref": [
                                        ],
                                        "sentiment": [
                                        ]
                                    },
                                    "date": "2019-02-12 16:41:23.142000"
                                },
                                {
                                    "id": "5c62f7330110b36bdd1dc5d8",
                                    "active_skill": "chitchat",
                                    "confidence": 0.85,
                                    "text": "Привет, я бот!",
                                    "user_id": "5c62f7330110b36bdd1dc5d6",
                                    "annotations": {
                                        "ner": [
                                        ],
                                        "coref": [
                                        ],
                                        "sentiment": [
                                        ]
                                    },
                                    "date": "2019-02-12 16:41:23.142000"
                                },
                                {
                                    "id": "5c62f7330110b36bdd1dc5d9",
                                    "text": "Как дела?",
                                    "user_id": "5c62f7330110b36bdd1dc5d5",
                                    "annotations": {
                                        "ner": [
                                        ],
                                        "coref": [
                                        ],
                                        "sentiment": [
                                        ]
                                    },
                                    "date": "2019-02-12 16:41:23.142000"
                                }
                            ],
                            'user': {
                                'id': '5c62f7330110b36bdd1dc5d5',
                                'user_telegram_id': '44d279ea-62ab-4c71-9adb-ed69143c12eb',
                                'user_type': 'human',
                                'device_type': None,
                                'personality': None
                            },
                            'bot': {
                                'id': '5c62f7330110b36bdd1dc5d6',
                                'user_telegram_id': '56f1d5b2-db1a-4128-993d-6cd1bc1b938f',
                                'user_type': 'bot',
                                'device_type': None,
                                'personality': None
                            },
                            'channel_type': 'telegram'
                        }
                    ]
                }
            }
        ],
        'responses': {
            "200": {
                "description": "A skill response",
                'example': {
                    'responses': [{name: 'sample-answer' for name in model.out_params}]
                }
            }
        }
    }

    @app.route('/')
    def index():
        return redirect('/apidocs/')

    @app.route(endpoint, methods=['POST'])
    @swag_from(endpoint_description)
    def answer():
        return interact_skill(model, batch_size)

    app.run(host=host, port=port, threaded=False, ssl_context=ssl_context)
