import sys
from pathlib import Path

from flask import Flask, request, jsonify, redirect
from flasgger import Swagger
from flask_cors import CORS

from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.common.log import get_logger


SERVER_CONFIG_FILENAME = 'server_config.json'

log = get_logger(__name__)

app = Flask(__name__)
Swagger(app)
CORS(app)


def init_model(model_config_path):
    config = read_json(model_config_path)
    model = build_model_from_config(config)
    model_name = type(model.get_main_component()).__name__
    return model, model_name


def get_server_params(config, model_name):
    server_params = config['common_defaults']

    if model_name in config['model_defaults']:
        model_defaults = config['model_defaults'][model_name]
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
    result = prediction[0]
    return jsonify(result), 200


def start_model_server(model_config_path):
    config_dir = Path(__file__).resolve().parent
    config_path = Path(config_dir, SERVER_CONFIG_FILENAME).resolve()
    config = read_json(config_path)

    model, model_name = init_model(model_config_path)

    server_params = get_server_params(config, model_name)
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
