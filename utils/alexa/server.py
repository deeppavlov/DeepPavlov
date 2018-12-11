from datetime import timedelta
from pathlib import Path
from queue import Queue
from typing import Union, Optional

from flask import Flask, request, jsonify, redirect
from flasgger import Swagger
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
                            stateful: bool = False, port: Optional[int] = None) -> None:

    def get_default_agent() -> DefaultAgent:
        model = build_model(model_config)
        skill = DefaultStatelessSkill(model)
        agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())
        return agent

    run_alexa_server(get_default_agent, multi_instance, stateful, port=port)


def run_alexa_server(agent_generator: callable, multi_instance: bool = False,
                     stateful: bool = False, port: Optional[int] = None) -> None:

    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME).resolve()
    server_params = read_json(server_config_path)

    host = server_params['common_defaults']['host']
    port = port or server_params['common_defaults']['port']

    alexa_server_params = server_params['alexa_defaults']

    alexa_server_params['multi_instance'] = multi_instance or server_params['common_defaults']['multi_instance']
    alexa_server_params['stateful'] = stateful or server_params['common_defaults']['stateful']
    alexa_server_params['amazon_cert_lifetime'] = AMAZON_CERTIFICATE_LIFETIME

    input_q = Queue()
    output_q = Queue()

    bot = Bot(agent_generator, alexa_server_params, input_q, output_q)
    bot.start()

    @app.route('/')
    def index():
        return redirect('/apidocs/')

    @app.route('/interact', methods=['POST'])
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

    app.run(host=host, port=port, threaded=True)
