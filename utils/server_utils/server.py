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
from pathlib import Path
from typing import List, Tuple

from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify, redirect, Response
from flask_cors import CORS

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.core.agent.dialog_logger import DialogLogger
from deeppavlov.core.data.utils import check_nested_dict_keys, jsonify_data

SERVER_CONFIG_FILENAME = 'server_config.json'

log = get_logger(__name__)

app = Flask(__name__)
Swagger(app)
CORS(app)

dialog_logger = DialogLogger(agent_name='dp_api')


def init_model(model_config_path):
    model_config = read_json(model_config_path)
    model = build_model(model_config)
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

    return server_params


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


def start_model_server(model_config_path, https=False, ssl_key=None, ssl_cert=None):
    server_config_path = get_settings_path() / SERVER_CONFIG_FILENAME
    server_params = get_server_params(server_config_path, model_config_path)

    host = server_params['host']
    port = server_params['port']
    model_endpoint = server_params['model_endpoint']
    model_args_names = server_params['model_args_names']

    https = https or server_params['https']

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

    model = init_model(model_config_path)

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
