import ssl
from logging import getLogger
from pathlib import Path
from queue import Queue
from typing import Union, Optional

from flasgger import Swagger
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS

from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.agents.processors.default_rich_content_processor import DefaultRichContentWrapper
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.skills.default_skill.default_skill import DefaultStatelessSkill
from utils.ms_bot_framework_utils.bot import Bot

SERVER_CONFIG_FILENAME = 'server_config.json'

AUTH_URL = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"
AUTH_HOST = "login.microsoftonline.com"
AUTH_CONTENT_TYPE = "application/x-www-form-urlencoded"
AUTH_GRANT_TYPE = "client_credentials"
AUTH_SCOPE = "https://api.botframework.com/.default"

log = getLogger(__name__)

app = Flask(__name__)
Swagger(app)
CORS(app)


def run_ms_bf_default_agent(model_config: Union[str, Path, dict],
                            app_id: str,
                            app_secret: str,
                            multi_instance: bool = False,
                            stateful: bool = False,
                            port: Optional[int] = None,
                            https: bool = False,
                            ssl_key: str = None,
                            ssl_cert: str = None,
                            default_skill_wrap: bool = True):

    def get_default_agent():
        model = build_model(model_config)
        skill = DefaultStatelessSkill(model) if default_skill_wrap else model
        agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())
        return agent

    run_ms_bot_framework_server(agent_generator=get_default_agent,
                                app_id=app_id,
                                app_secret=app_secret,
                                multi_instance=multi_instance,
                                stateful=stateful,
                                port=port,
                                https=https,
                                ssl_key=ssl_key,
                                ssl_cert=ssl_cert)


def run_ms_bot_framework_server(agent_generator: callable,
                                app_id: str,
                                app_secret: str,
                                multi_instance: bool = False,
                                stateful: bool = False,
                                port: Optional[int] = None,
                                https: bool = False,
                                ssl_key: str = None,
                                ssl_cert: str = None):

    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME).resolve()
    server_params = read_json(server_config_path)

    host = server_params['common_defaults']['host']
    port = port or server_params['common_defaults']['port']

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

    app.run(host=host, port=port, threaded=True, ssl_context=ssl_context)
