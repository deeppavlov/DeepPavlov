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

import threading
from collections import namedtuple
from datetime import datetime, timedelta
from logging import getLogger
from pathlib import Path
from queue import Empty, Queue
from threading import Thread, Timer
from typing import Dict, Optional, Union

import requests
import telebot
from OpenSSL.crypto import X509
from requests.exceptions import HTTPError

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.utils.connector.conversation import AlexaConversation, AliceConversation, BaseConversation
from deeppavlov.utils.connector.conversation import MSConversation, TelegramConversation
from deeppavlov.utils.connector.ssl_tools import verify_cert, verify_signature

CONNECTOR_CONFIG_FILENAME = 'server_config.json'
INPUT_QUEUE_TIMEOUT = 1

log = getLogger(__name__)

ValidatedCert = namedtuple('ValidatedCert', ['cert', 'expiration_timestamp'])


class BaseBot(Thread):
    """Routes requests to conversations, sends responses to channel.

    Attributes:
        input_queue: Queue for incoming requests from the channel.

    """
    input_queue: Queue
    _run_flag: bool
    _model: Chainer
    _conversations: Dict[str, BaseConversation]

    def __init__(self,
                 model_config: Union[str, Path, dict],
                 input_queue: Queue) -> None:
        """Builds DeepPavlov model, initiates class attributes.

        Args:
            model_config: Path to DeepPavlov model config file.
            input_queue: Queue for incoming requests from channel.

        """
        super(BaseBot, self).__init__()
        self.input_queue = input_queue
        self._run_flag = True
        self._model = build_model(model_config)
        self._conversations = dict()
        log.info('Bot initiated')

    def run(self) -> None:
        """Thread run method implementation. Routes requests from ``input_queue`` to request handler."""
        while self._run_flag:
            try:
                request: dict = self.input_queue.get(timeout=INPUT_QUEUE_TIMEOUT)
            except Empty:
                pass
            else:
                response = self._handle_request(request)
                self._send_response(response)

    def join(self, timeout: Optional[float] = None) -> None:
        """Thread join method implementation. Stops reading requests from ``input_queue``, cancels all timers.

        Args:
            timeout: Timeout for join operation in seconds. If the timeout argument is not present or None,
                the operation will block until the thread terminates.

        """
        self._run_flag = False
        for timer in threading.enumerate():
            if isinstance(timer, Timer):
                timer.cancel()
        Thread.join(self, timeout)

    def _del_conversation(self, conversation_key: Union[int, str]) -> None:
        """Deletes Conversation instance.

        Args:
            conversation_key: Conversation key.

        """
        if conversation_key in self._conversations.keys():
            del self._conversations[conversation_key]
            log.info(f'Deleted conversation, key: {conversation_key}')

    def _handle_request(self, request: dict) -> Optional[dict]:
        """Routes the request to the appropriate conversation.

        Args:
            request: Request from the channel.

        Returns:
            response: Corresponding response to the channel request if replies are sent via bot, None otherwise.

        """
        raise NotImplementedError

    def _send_response(self, response: Optional[dict]) -> None:
        """Sends response to the request back to the channel.

        Args:
            response: Corresponding response to the channel request if replies are sent via bot, None otherwise.

        """
        raise NotImplementedError

    def _get_connector_params(self) -> dict:
        """Reads bot and conversation default params from connector config file.

         Returns:
             connector_defaults: Dictionary containing bot defaults and conversation defaults dicts.

        """
        connector_config_path = get_settings_path() / CONNECTOR_CONFIG_FILENAME
        connector_config: dict = read_json(connector_config_path)

        bot_name = type(self).__name__
        conversation_defaults = connector_config['telegram']
        bot_defaults = connector_config['deprecated'].get(bot_name, conversation_defaults)

        connector_defaults = {'bot_defaults': bot_defaults,
                              'conversation_defaults': conversation_defaults}

        return connector_defaults


class AlexaBot(BaseBot):
    """Validates Alexa requests and routes them to conversations, sends responses to Alexa.

    Attributes:
        input_queue: Queue for incoming requests from Alexa.
        output_queue: Queue for outgoing responses to Alexa.

    """
    output_queue: Queue
    _conversation_config: dict
    _amazon_cert_lifetime: timedelta
    _request_timestamp_tolerance_secs: int
    _refresh_valid_certs_period_secs: int
    _valid_certificates: Dict[str, ValidatedCert]
    _timer: Timer

    def __init__(self,
                 model_config: Union[str, Path, dict],
                 input_queue: Queue,
                 output_queue: Queue) -> None:
        """Initiates class attributes.

        Args:
            model_config: Path to DeepPavlov model config file.
            input_queue: Queue for incoming requests from Alexa.
            output_queue: Queue for outgoing responses to Alexa.

        """
        super(AlexaBot, self).__init__(model_config, input_queue)
        self.output_queue = output_queue

        connector_config: dict = self._get_connector_params()
        self._conversation_config: dict = connector_config['conversation_defaults']
        bot_config: dict = connector_config['bot_defaults']

        self._conversation_config['intent_name'] = bot_config['intent_name']
        self._conversation_config['slot_name'] = bot_config['slot_name']

        self._amazon_cert_lifetime = timedelta(seconds=bot_config['amazon_cert_lifetime_secs'])
        self._request_timestamp_tolerance_secs = bot_config['request_timestamp_tolerance_secs']
        self._refresh_valid_certs_period_secs = bot_config['refresh_valid_certs_period_secs']
        self._valid_certificates = {}
        self._refresh_valid_certs()

    def _refresh_valid_certs(self) -> None:
        """Provides cleanup of periodical certificates with expired validation."""
        self._timer = Timer(self._refresh_valid_certs_period_secs, self._refresh_valid_certs)
        self._timer.start()

        expired_certificates = []

        for valid_cert_url, valid_cert in self._valid_certificates.items():
            valid_cert: ValidatedCert = valid_cert
            cert_expiration_time: datetime = valid_cert.expiration_timestamp
            if datetime.utcnow() > cert_expiration_time:
                expired_certificates.append(valid_cert_url)

        for expired_cert_url in expired_certificates:
            del self._valid_certificates[expired_cert_url]
            log.info(f'Validation period of {expired_cert_url} certificate expired')

    def _verify_request(self, signature_chain_url: str, signature: str, request_body: bytes) -> bool:
        """Provides series of Alexa request verifications against Amazon Alexa requirements.

        Args:
            signature_chain_url: Signature certificate URL from SignatureCertChainUrl HTTP header.
            signature: Base64 decoded Alexa request signature from Signature HTTP header.
            request_body: full HTTPS request body

        Returns:
            result: True if verification was successful, False otherwise.

        """
        if signature_chain_url not in self._valid_certificates.keys():
            amazon_cert: X509 = verify_cert(signature_chain_url)
            if amazon_cert:
                expiration_timestamp = datetime.utcnow() + self._amazon_cert_lifetime
                validated_cert = ValidatedCert(cert=amazon_cert, expiration_timestamp=expiration_timestamp)
                self._valid_certificates[signature_chain_url] = validated_cert
                log.info(f'Certificate {signature_chain_url} validated')
            else:
                log.error(f'Certificate {signature_chain_url} validation failed')
                return False
        else:
            validated_cert: ValidatedCert = self._valid_certificates[signature_chain_url]
            amazon_cert: X509 = validated_cert.cert

        if verify_signature(amazon_cert, signature, request_body):
            result = True
        else:
            log.error(f'Failed signature verification for request: {request_body.decode("utf-8", "replace")}')
            result = False

        return result

    def _handle_request(self, request: dict) -> dict:
        """Processes Alexa request and returns response.

        Args:
            request: Dict with Alexa request payload and metadata.

        Returns:
            result: Alexa formatted or error response.

        """
        request_body: bytes = request['request_body']
        signature_chain_url: str = request['signature_chain_url']
        signature: str = request['signature']
        alexa_request: dict = request['alexa_request']

        if not self._verify_request(signature_chain_url, signature, request_body):
            return {'error': 'failed certificate/signature check'}

        timestamp_str = alexa_request['request']['timestamp']
        timestamp_datetime = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
        now = datetime.utcnow()

        delta = now - timestamp_datetime if now >= timestamp_datetime else timestamp_datetime - now

        if abs(delta.seconds) > self._request_timestamp_tolerance_secs:
            log.error(f'Failed timestamp check for request: {request_body.decode("utf-8", "replace")}')
            return {'error': 'failed request timestamp check'}

        conversation_key = alexa_request['session']['sessionId']

        if conversation_key not in self._conversations:
            self._conversations[conversation_key] = \
                AlexaConversation(config=self._conversation_config,
                                  model=self._model,
                                  self_destruct_callback=self._del_conversation,
                                  conversation_id=conversation_key)

            log.info(f'Created new conversation, key: {conversation_key}')

        conversation = self._conversations[conversation_key]
        response = conversation.handle_request(alexa_request)

        return response

    def _send_response(self, response: dict) -> None:
        """Sends response to Alexa.

        Args:
            response: Alexa formatted or error response.

        """
        self.output_queue.put(response)


class AliceBot(BaseBot):
    """Processes Alice requests and routes them to conversations, returns responses to Alice.

    Attributes:
        input_queue: Queue for incoming requests from Alice.
        output_queue: Queue for outgoing responses to Alice.

    """
    output_queue: Queue
    _conversation_config: dict

    def __init__(self,
                 model_config: Union[str, Path, dict],
                 input_queue: Queue,
                 output_queue: Queue) -> None:
        """Initiates class attributes.

        Args:
            model_config: Path to DeepPavlov model config file.
            input_queue: Queue for incoming requests from Alice.
            output_queue: Queue for outgoing responses to Alice.

        """
        super(AliceBot, self).__init__(model_config, input_queue)
        self.output_queue = output_queue
        connector_config: dict = self._get_connector_params()
        self._conversation_config = connector_config['conversation_defaults']

    def _handle_request(self, request: dict) -> dict:
        """Processes Alice request and returns response.

        Args:
            request: Dict with Alice request payload and metadata.

        Returns:
            result: Alice formatted response.

        """
        conversation_key = request['session']['session_id']

        if conversation_key not in self._conversations:
            self._conversations[conversation_key] = \
                AliceConversation(config=self._conversation_config,
                                  model=self._model,
                                  self_destruct_callback=self._del_conversation,
                                  conversation_id=conversation_key)
            log.info(f'Created new conversation, key: {conversation_key}')
        conversation = self._conversations[conversation_key]
        response = conversation.handle_request(request)

        return response

    def _send_response(self, response: dict) -> None:
        """Sends response to Alice.

        Args:
            response: Alice formatted response.

        """
        self.output_queue.put(response)


class MSBot(BaseBot):
    """Routes Microsoft Bot Framework requests to conversations, sends responses to Bot Framework.

    Attributes:
        input_queue: Queue for incoming requests from Microsoft Bot Framework.

    """
    _conversation_config: dict
    _auth_polling_interval: int
    _auth_url: str
    _auth_headers: dict
    _auth_payload: dict
    _http_session: requests.Session

    def __init__(self,
                 model_config: Union[str, Path, dict],
                 input_queue: Queue,
                 client_id: Optional[str],
                 client_secret: Optional[str]) -> None:
        """Initiates class attributes.

        Args:
            model_config: Path to DeepPavlov model config file.
            input_queue: Queue for incoming requests from Microsoft Bot Framework.
            client_id: Microsoft App ID.
            client_secret: Microsoft App Secret.

        Raises:
            ValueError: If ``client_id`` or ``client_secret`` were not set neither in the configuration file nor
                in method arguments.

        """
        super(MSBot, self).__init__(model_config, input_queue)
        connector_config: dict = self._get_connector_params()
        bot_config: dict = connector_config['bot_defaults']
        bot_config['auth_payload']['client_id'] = client_id or bot_config['auth_payload']['client_id']
        bot_config['auth_payload']['client_secret'] = client_secret or bot_config['auth_payload']['client_secret']

        if not bot_config['auth_payload']['client_id']:
            e = ValueError('Microsoft Bot Framework app id required: initiate -i param '
                           'or auth_payload.client_id param in server configuration file')
            log.error(e)
            raise e

        if not bot_config['auth_payload']['client_secret']:
            e = ValueError('Microsoft Bot Framework app secret required: initiate -s param '
                           'or auth_payload.client_secret param in server configuration file')
            log.error(e)
            raise e

        self._conversation_config = connector_config['conversation_defaults']
        self._auth_polling_interval = bot_config['auth_polling_interval']
        self._auth_url = bot_config['auth_url']
        self._auth_headers = bot_config['auth_headers']
        self._auth_payload = bot_config['auth_payload']
        self._http_session = requests.Session()
        self._update_access_info()

    def _update_access_info(self) -> None:
        """Updates headers for http_session used to send responses to Bot Framework.

        Raises:
            HTTPError: If authentication token request returned other than 200 status code.

        """
        self._timer = threading.Timer(self._auth_polling_interval, self._update_access_info)
        self._timer.start()

        result = requests.post(url=self._auth_url,
                               headers=self._auth_headers,
                               data=self._auth_payload)

        status_code = result.status_code
        if status_code != 200:
            raise HTTPError(f'Authentication token request returned wrong HTTP status code: {status_code}')

        access_info = result.json()
        headers = {
            'Authorization': f"{access_info['token_type']} {access_info['access_token']}",
            'Content-Type': 'application/json'
        }

        self._http_session.headers.update(headers)

        log.info(f'Obtained authentication information from Microsoft Bot Framework: {str(access_info)}')

    def _handle_request(self, request: dict) -> None:
        """Routes MS Bot Framework request to conversation.

        Args:
            request: Dict with MS Bot Framework request payload and metadata.

        """
        conversation_key = request['conversation']['id']

        if conversation_key not in self._conversations:
            self._conversations[conversation_key] = \
                MSConversation(config=self._conversation_config,
                               model=self._model,
                               self_destruct_callback=self._del_conversation,
                               conversation_id=conversation_key,
                               http_session=self._http_session)

            log.info(f'Created new conversation, key: {conversation_key}')

        conversation = self._conversations[conversation_key]
        conversation.handle_request(request)

    def _send_response(self, response: dict) -> None:
        """Dummy method to match ``run`` method body."""
        pass


class TelegramBot(BaseBot):
    """Routes messages from Telegram to conversations, sends responses back."""
    _conversation_config: dict
    _token: str

    def __init__(self, model_config: Union[str, Path, dict], token: Optional[str]) -> None:
        """Initiates and validates class attributes.

        Args:
            model_config: Path to DeepPavlov model config file.
            token: Telegram bot token.

        Raises:
            ValueError: If telegram token was not set neither in config file nor in method arguments.

        """
        super(TelegramBot, self).__init__(model_config, Queue())
        connector_config: dict = self._get_connector_params()
        bot_config: dict = connector_config['bot_defaults']
        self._conversation_config = connector_config['conversation_defaults']
        self._token = token or bot_config['token']

        if not self._token:
            e = ValueError('Telegram token required: initiate -t param or telegram_defaults/token '
                           'in server configuration file')
            log.error(e)
            raise e

    def start(self) -> None:
        """Starts polling messages from Telegram, routes messages to handlers."""
        bot = telebot.TeleBot(self._token)
        bot.remove_webhook()

        @bot.message_handler(commands=['start'])
        def send_start_message(message: telebot.types.Message) -> None:
            chat_id = message.chat.id
            out_message = self._conversation_config['start_message']
            bot.send_message(chat_id, out_message)

        @bot.message_handler(commands=['help'])
        def send_help_message(message: telebot.types.Message) -> None:
            chat_id = message.chat.id
            out_message = self._conversation_config['help_message']
            bot.send_message(chat_id, out_message)

        @bot.message_handler()
        def handle_inference(message: telebot.types.Message) -> None:
            chat_id = message.chat.id
            context = message.text

            if chat_id not in self._conversations:
                self._conversations[chat_id] = \
                    TelegramConversation(config=self._conversation_config,
                                         model=self._model,
                                         self_destruct_callback=self._del_conversation,
                                         conversation_id=chat_id)

            conversation = self._conversations[chat_id]
            response = conversation.handle_request(context)
            bot.send_message(chat_id, response)

        bot.polling()

    def _handle_request(self, request: dict) -> None:
        """Dummy method to match ``run`` method body."""
        pass

    def _send_response(self, response: Optional[dict]) -> None:
        """Dummy method to match ``run`` method body."""
        pass
