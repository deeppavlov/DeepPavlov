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

import sys
from pathlib import Path

from flask import Flask, request, jsonify, redirect
from flasgger import Swagger, swag_from
from flask_cors import CORS

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.data.utils import check_nested_dict_keys, jsonify_data
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component

SERVER_CONFIG_FILENAME = 'server_config.json'

log = get_logger(__name__)

app = Flask(__name__)
Swagger(app)
CORS(app)


def init_model(model_config_path):
    model_config = read_json(model_config_path)
    model = build_model_from_config(model_config)
    return model


def get_server_params(server_config_path, model_config_path):
    server_config = read_json(server_config_path)
    model_config = read_json(model_config_path)

    server_params = server_config['common_defaults']

    if check_nested_dict_keys(model_config, ['metadata', 'labels', 'server_utils']):
        model_tag = model_config['metadata']['labels']['server_utils']
        if model_tag in server_config['model_defaults']:
            model_defaults = server_config['model_defaults'][model_tag]
            for param_name in model_defaults.keys():
                if model_defaults[param_name]:
                    server_params[param_name] = model_defaults[param_name]

    for param_name in server_params.keys():
        if not server_params[param_name]:
            log.error('"{}" parameter should be set either in common_defaults '
                      'or in model_defaults section of {}'.format(param_name, SERVER_CONFIG_FILENAME))
            sys.exit(1)

    return server_params


memory = {}


def interact_alice(model: Component, params_names: list):
    """
    Exchange messages between basic pipelines and the Yandex.Dialogs service.
    If the pipeline returns multiple values, only the first one is forwarded to Yandex.
    """
    data = request.get_json()
    text = data['request']['command'].strip()

    session_id = data['session']['session_id']
    message_id = data['session']['message_id']
    user_id = data['session']['user_id']

    response = {
        'response': {
            'end_session': True
        },
        "session": {
            'session_id': session_id,
            'message_id': message_id,
            'user_id': user_id
        },
        'version': '1.0'
    }

    params = memory.pop(session_id, [])
    if text:
        params += [text]

    if len(params) < len(params_names):
        memory[session_id] = params
        response['response']['text'] = 'Пожалуйста, введите параметр ' + params_names[len(params)]
        response['response']['end_session'] = False
        return jsonify(response), 200

    if len(params) == 1:
        params = params[0]

    response_text = model([params])[0]
    if not isinstance(response_text, str) and isinstance(response_text, (list, tuple)):
        try:
            response_text = response_text[0]
        except Exception as e:
            log.warning(f'Could not get the first element of `{repr(response_text)}` because of `{e}`')

    response['response']['text'] = str(response_text)
    return jsonify(response), 200


def interact(model, params_names):
    if not request.is_json:
        log.error("request Content-Type header is not application/json")
        return jsonify({
            "error": "request Content-Type header is not application/json"
        }), 400

    model_args = []

    data = request.get_json()
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

    if len(params_names) == 1:
        model_args = model_args[0]
    else:
        batch_size = list(lengths)[0]
        model_args = [arg or [None] * batch_size for arg in model_args]
        model_args = list(zip(*model_args))

    prediction = model(model_args)
    result = jsonify_data(prediction)
    return jsonify(result), 200


def start_model_server(model_config_path, alice=False):
    server_config_dir = Path(__file__).parent
    server_config_path = server_config_dir.parent / SERVER_CONFIG_FILENAME

    model = init_model(model_config_path)

    server_params = get_server_params(server_config_path, model_config_path)
    host = server_params['host']
    port = server_params['port']
    model_endpoint = server_params['model_endpoint']
    model_args_names = server_params['model_args_names']

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

    if alice:
        endpoint_description['parameters'][0]['example'] = {
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

    @app.route(model_endpoint, methods=['POST'])
    @swag_from(endpoint_description)
    def answer():
        return interact_alice(model, model_args_names) if alice else interact(model, model_args_names)

    app.run(host=host, port=port, threaded=False)
