from pathlib import Path
from queue import Queue

from flask import Flask, request, jsonify, redirect
from flasgger import Swagger
from flask_cors import CORS

from utils.ms_bot_framework_utils.bot import Bot
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json

SERVER_CONFIG_FILENAME = 'server_config.json'

log = get_logger(__name__)

app = Flask(__name__)
Swagger(app)
CORS(app)


def start_bot_framework_server(model_config_path: str, app_id: str, app_secret: str):
    server_config_dir = Path(__file__).resolve().parent
    server_config_path = Path(server_config_dir, '..', SERVER_CONFIG_FILENAME).resolve()
    server_params = read_json(server_config_path)

    host = server_params['common_defaults']['host']
    port = server_params['common_defaults']['port']

    input_q = Queue()
    bot = Bot(server_params, model_config_path, app_id, app_secret, input_q)
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
