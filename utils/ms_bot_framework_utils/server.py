from pathlib import Path
from queue import Queue
from typing import Union

from flask import Flask, request, jsonify, redirect
from flasgger import Swagger
from flask_cors import CORS

from utils.ms_bot_framework_utils.bot import Bot
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.default_rich_content_processor import DefaultRichContentWrapper
from deeppavlov.skills.default_skill.default_skill import DefaultStatelessSkill

SERVER_CONFIG_FILENAME = 'server_config.json'

AUTH_URL = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
AUTH_HOST = "login.microsoftonline.com"
AUTH_CONTENT_TYPE = "application/x-www-form-urlencoded"
AUTH_GRANT_TYPE = "client_credentials"
AUTH_SCOPE = "https://api.botframework.com/.default"

log = get_logger(__name__)

app = Flask(__name__)
Swagger(app)
CORS(app)


def run_ms_bf_default_agent(model_config: Union[str, Path, dict], app_id: str, app_secret: str,
                            multi_instance: bool = False, stateful: bool = False):
    def get_default_agent():
        model = build_model(model_config)
        skill = DefaultStatelessSkill(model)
        agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())
        return agent

    run_ms_bot_framework_server(get_default_agent, app_id, app_secret, multi_instance, stateful)


def run_ms_bot_framework_server(agent_generator: callable, app_id: str, app_secret: str,
                                multi_instance: bool = False, stateful: bool = False):

    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME).resolve()
    server_params = read_json(server_config_path)

    host = server_params['common_defaults']['host']
    port = server_params['common_defaults']['port']

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

    input_q = Queue()
    bot = Bot(agent_generator, ms_bf_server_params, input_q)
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
