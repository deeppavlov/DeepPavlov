import sys
from pathlib import Path

from flask import Flask, request, jsonify, redirect
from flasgger import Swagger
from flask_cors import CORS

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.data.utils import check_nested_dict_keys, jsonify_data
from deeppavlov.core.common.log import get_logger


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


def interact(model, params_names):
    if not request.is_json:
        return jsonify({
            "error": "request must contains json data"
        }), 400

    model_args = []

    for param_name in params_names:
        param_value = request.get_json().get(param_name)
        model_args.append(param_value)
    if len(params_names) > 1:
        model_args = [model_args]

    prediction = model(model_args)
    result = jsonify_data(prediction[0])
    return jsonify(result), 200


def start_model_server(model_config_path):
    server_config_dir = Path(__file__).resolve().parent
    server_config_path = Path(server_config_dir, SERVER_CONFIG_FILENAME).resolve()

    model = init_model(model_config_path)

    server_params = get_server_params(server_config_path, model_config_path)
    host = server_params['host']
    port = server_params['port']
    model_endpoint = server_params['model_endpoint']
    model_args_names = server_params['model_args_names']

    @app.route('/')
    def index():
        return redirect('/apidocs/')

    @app.route(model_endpoint, methods=['POST'])
    def answer_intents():
        """
        Skill
        ---
        parameters:
         - name: data
           in: body
           required: true
           type: json
        """
        return interact(model, model_args_names)

    app.run(host=host, port=port)
