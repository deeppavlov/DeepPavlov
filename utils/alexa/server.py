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
from datetime import timedelta
from pathlib import Path
from queue import Queue
from typing import Union, Optional

from flask import Flask, request, jsonify, redirect
from flasgger import Swagger, swag_from
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
                            stateful: bool = False, port: Optional[int] = None, https: bool = False,
                            ssl_key: str = None, ssl_cert: str = None) -> None:
    """Creates Alexa agents factory and initiates Alexa web service.

    Wrapper around run_alexa_server. Allows raise Alexa web service with
    DeepPavlov config in backend.

    Args:
        model_config: DeepPavlov config path.
        multi_instance: Multi instance mode flag.
        stateful: Stateful mode flag.
        port: Flask web service port.
        https: Flag for running Alexa skill service in https mode.
        ssl_key: SSL key file path.
        ssl_cert: SSL certificate file path.
    """

    def get_default_agent() -> DefaultAgent:
        model = build_model(model_config)
        skill = DefaultStatelessSkill(model)
        agent = DefaultAgent([skill], skills_processor=DefaultRichContentWrapper())
        return agent

    run_alexa_server(agent_generator=get_default_agent,
                     multi_instance=multi_instance,
                     stateful=stateful,
                     port=port,
                     https=https,
                     ssl_key=ssl_key,
                     ssl_cert=ssl_cert)


def run_alexa_server(agent_generator: callable, multi_instance: bool = False,
                     stateful: bool = False, port: Optional[int] = None, https: bool = False,
                     ssl_key: str = None, ssl_cert: str = None) -> None:
    """Initiates Flask web service with Alexa skill.

    Args:
        agent_generator: Callback Alexa agents factory.
        multi_instance: Multi instance mode flag.
        stateful: Stateful mode flag.
        port: Flask web service port.
        https: Flag for running Alexa skill service in https mode.
        ssl_key: SSL key file path.
        ssl_cert: SSL certificate file path.
    """
    server_config_path = Path(get_settings_path(), SERVER_CONFIG_FILENAME).resolve()
    server_params = read_json(server_config_path)

    host = server_params['common_defaults']['host']
    port = port or server_params['common_defaults']['port']

    alexa_server_params = server_params['alexa_defaults']

    alexa_server_params['multi_instance'] = multi_instance or server_params['common_defaults']['multi_instance']
    alexa_server_params['stateful'] = stateful or server_params['common_defaults']['stateful']
    alexa_server_params['amazon_cert_lifetime'] = AMAZON_CERTIFICATE_LIFETIME

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
    output_q = Queue()

    bot = Bot(agent_generator, alexa_server_params, input_q, output_q)
    bot.start()

    endpoint_description = {
        'description': 'Amazon Alexa custom service endpoint',
        'parameters': [
            {
                'name': 'Signature',
                'in': 'header',
                'required': 'true',
                'type': 'string',
                'example': 'Z5H5wqd06ExFVPNfJiqhKvAFjkf+cTVodOUirucHGcEVAMO1LfvgqWUkZ/X1ITDZbI0w+SMwVkEQZlkeThbVS/54M22StNDUtfz4Ua20xNDpIPwcWIACAmZ38XxbbTEFJI5WwqrbilNcfzqiGrIPfdO5rl+/xUjHFUdcJdUY/QzBxXsceytVYfEiR9MzOCN2m4C0XnpThUavAu159KrLj8AkuzN0JF87iXv+zOEeZRgEuwmsAnJrRUwkJ4yWokEPnSVdjF0D6f6CscfyvRe9nsWShq7/zRTa41meweh+n006zvf58MbzRdXPB22RI4AN0ksWW7hSC8/QLAKQE+lvaw==',
            },
            {
                'name': 'Signaturecertchainurl',
                'in': 'header',
                'required': 'true',
                'type': 'string',
                'example': 'https://s3.amazonaws.com/echo.api/echo-api-cert-6-ats.pem',
            },
            {
                'name': 'data',
                'in': 'body',
                'required': 'true',
                'example': {
                    'version': '1.0',
                    'session': {
                        'new': False,
                        'sessionId': 'amzn1.echo-api.session.3c6ebffd-55b9-4e1a-bf3c-c921c1801b63',
                        'application': {
                            'applicationId': 'amzn1.ask.skill.8b17a5de-3749-4919-aa1f-e0bbaf8a46a6'
                        },
                        'attributes': {
                            'sessionId': 'amzn1.echo-api.session.3c6ebffd-55b9-4e1a-bf3c-c921c1801b63'
                        },
                        'user': {
                            'userId': 'amzn1.ask.account.AGR4R2LOVHMNMNOGROBVNLU7CL4C57X465XJF2T2F55OUXNTLCXDQP3I55UXZIALEKKZJ6Q2MA5MEFSMZVPEL5NVZS6FZLEU444BVOLPB5WVH5CHYTQAKGD7VFLGPRFZVHHH2NIB4HKNHHGX6HM6S6QDWCKXWOIZL7ONNQSBUCVPMZQKMCYXRG5BA2POYEXFDXRXCGEVDWVSMPQ'
                        }
                    },
                    'context': {
                        'System': {
                            'application': {
                                'applicationId': 'amzn1.ask.skill.8b17a5de-3749-4919-aa1f-e0bbaf8a46a6'
                            },
                            'user': {
                                'userId': 'amzn1.ask.account.AGR4R2LOVHMNMNOGROBVNLU7CL4C57X465XJF2T2F55OUXNTLCXDQP3I55UXZIALEKKZJ6Q2MA5MEFSMZVPEL5NVZS6FZLEU444BVOLPB5WVH5CHYTQAKGD7VFLGPRFZVHHH2NIB4HKNHHGX6HM6S6QDWCKXWOIZL7ONNQSBUCVPMZQKMCYXRG5BA2POYEXFDXRXCGEVDWVSMPQ'
                            },
                            'device': {
                                'deviceId': 'amzn1.ask.device.AFQAMLYOYQUUACSE7HFVYS4ZI2KUB35JPHQRUPKTDCAU3A47WESP5L57KSWT5L6RT3FVXWH4OA2DNPJRMZ2VGEIACF3PJEIDCOUWUBC4W5RPJNUB3ZVT22J4UJN5UL3T2UBP36RVHFJ5P4IPT2HUY3P2YOY33IOU4O33HUAG7R2BUNROEH4T2',
                                'supportedInterfaces': {}
                            },
                            'apiEndpoint': 'https://api.amazonalexa.com',
                            'apiAccessToken': 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6IjEifQ.eyJhdWQiOiJodHRwczovL2FwaS5hbWF6b25hbGV4YS5jb20iLCJpc3MiOiJBbGV4YVNraWxsS2l0Iiwic3ViIjoiYW16bjEuYXNrLnNraWxsLjhiMTdhNWRlLTM3NDktNDkxOS1hYTFmLWUwYmJhZjhhNDZhNiIsImV4cCI6MTU0NTIyMzY1OCwiaWF0IjoxNTQ1MjIwMDU4LCJuYmYiOjE1NDUyMjAwNTgsInByaXZhdGVDbGFpbXMiOnsiY29uc2VudFRva2VuIjpudWxsLCJkZXZpY2VJZCI6ImFtem4xLmFzay5kZXZpY2UuQUZRQU1MWU9ZUVVVQUNTRTdIRlZZUzRaSTJLVUIzNUpQSFFSVVBLVERDQVUzQTQ3V0VTUDVMNTdLU1dUNUw2UlQzRlZYV0g0T0EyRE5QSlJNWjJWR0VJQUNGM1BKRUlEQ09VV1VCQzRXNVJQSk5VQjNaVlQyMko0VUpONVVMM1QyVUJQMzZSVkhGSjVQNElQVDJIVVkzUDJZT1kzM0lPVTRPMzNIVUFHN1IyQlVOUk9FSDRUMiIsInVzZXJJZCI6ImFtem4xLmFzay5hY2NvdW50LkFHUjRSMkxPVkhNTk1OT0dST0JWTkxVN0NMNEM1N1g0NjVYSkYyVDJGNTVPVVhOVExDWERRUDNJNTVVWFpJQUxFS0taSjZRMk1BNU1FRlNNWlZQRUw1TlZaUzZGWkxFVTQ0NEJWT0xQQjVXVkg1Q0hZVFFBS0dEN1ZGTEdQUkZaVkhISDJOSUI0SEtOSEhHWDZITTZTNlFEV0NLWFdPSVpMN09OTlFTQlVDVlBNWlFLTUNZWFJHNUJBMlBPWUVYRkRYUlhDR0VWRFdWU01QUSJ9fQ.jcomYhBhU485T4uoe2NyhWnL-kZHoPQKpcycFqa-1sy_lSIitfFGup9DKrf2NkN-I9lZ3xwq9llqx9WRN78fVJjN6GLcDhBDH0irPwt3n9_V7_5bfB6KARv5ZG-JKOmZlLBqQbnln0DAJ10D8HNiytMARNEwduMBVDNK0A5z6YxtRcLYYFD2-Ieg_V8Qx90eE2pd2U5xOuIEL0pXfSoiJ8vpxb8BKwaMO47tdE4qhg_k7v8ClwyXg3EMEhZFjixYNqdW1tCrwDGj58IWMXDyzZhIlRMh6uudMOT6scSzcNVD0v42IOTZ3S_X6rG01B7xhUDlZXMqkrCuzOyqctGaPw'
                        },
                        'Viewport': {
                            'experiences': [
                                {
                                    'arcMinuteWidth': 246,
                                    'arcMinuteHeight': 144,
                                    'canRotate': False,
                                    'canResize': False
                                }
                            ],
                            'shape': 'RECTANGLE',
                            'pixelWidth': 1024,
                            'pixelHeight': 600,
                            'dpi': 160,
                            'currentPixelWidth': 1024,
                            'currentPixelHeight': 600,
                            'touch': [
                                'SINGLE'
                            ]
                        }
                    },
                    'request': {
                        'type': 'IntentRequest',
                        'requestId': 'amzn1.echo-api.request.388d0f6e-04b9-4450-a687-b9abaa73ac6a',
                        'timestamp': '2018-12-19T11:47:38Z',
                        'locale': 'en-US',
                        'intent': {
                            'name': 'AskDeepPavlov',
                            'confirmationStatus': 'NONE',
                            'slots': {
                                'raw_input': {
                                    'name': 'raw_input',
                                    'value': 'my beautiful sandbox skill',
                                    'resolutions': {
                                        'resolutionsPerAuthority': [
                                            {
                                                'authority': 'amzn1.er-authority.echo-sdk.amzn1.ask.skill.8b17a5de-3749-4919-aa1f-e0bbaf8a46a6.GetInput',
                                                'status': {
                                                    'code': 'ER_SUCCESS_NO_MATCH'
                                                }
                                            }
                                        ]
                                    },
                                    'confirmationStatus': 'NONE',
                                    'source': 'USER'
                                }
                            }
                        }
                    }
                }
            }
        ],
        'responses': {
            "200": {
                "description": "A model response"
            }
        }
    }

    @app.route('/')
    def index():
        return redirect('/apidocs/')

    @app.route('/interact', methods=['POST'])
    @swag_from(endpoint_description)
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

    app.run(host=host, port=port, threaded=True, ssl_context=ssl_context)
