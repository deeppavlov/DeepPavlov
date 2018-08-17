from pathlib import Path
from queue import Queue

from flask import Flask, request, jsonify, redirect
from flasgger import Swagger
from flask_cors import CORS

from utils.ms_bot_framework_utils.bot import Bot
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json

SERVER_CONFIG_FILENAME = 'server_config.json'

AUTH_URL =  "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
AUTH_HOST = "login.microsoftonline.com"
AUTH_CONTENT_TYPE = "application/x-www-form-urlencoded"
AUTH_GRANT_TYPE = "client_credentials"
AUTH_SCOPE = "https://api.botframework.com/.default"

log = get_logger(__name__)

app = Flask(__name__)
Swagger(app)
CORS(app)


def start_bot_framework_server(model_config_path: str, app_id: str, app_secret: str,
                               multi_instance: bool = False, stateful: bool = False, use_history: bool = False):

    server_config_dir = Path(__file__).resolve().parent
    server_config_path = Path(server_config_dir, '..', SERVER_CONFIG_FILENAME).resolve()
    server_params = read_json(server_config_path)

    host = server_params['common_defaults']['host']
    port = server_params['common_defaults']['port']

    server_params = server_params['ms_bot_framework_defaults']

    server_params['model_config_path'] = model_config_path

    server_params['auth_url'] = AUTH_URL
    server_params['auth_host'] = AUTH_HOST
    server_params['auth_content_type'] = AUTH_CONTENT_TYPE
    server_params['auth_grant_type'] = AUTH_GRANT_TYPE
    server_params['auth_scope'] = AUTH_SCOPE

    server_params['auth_app_id'] = app_id
    server_params['auth_app_secret'] = app_secret

    server_params['multi_instance'] = multi_instance
    server_params['stateful'] = stateful
    server_params['use_history'] = use_history

    input_q = Queue()
    bot = Bot(server_params, input_q)
    bot.start()

    @app.route('/')
    def index():
        return redirect('/apidocs/')

    @app.route('/v3/conversations', methods=['POST'])
    def handle_activity():
        activity = request.get_json()
        bot.input_queue.put(activity)
        return jsonify({}), 200

    app.run(host=host, port=port, threaded=True)
