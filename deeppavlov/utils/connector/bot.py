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
from datetime import timedelta, datetime
from logging import getLogger
from pathlib import Path
from queue import Empty, Queue
from threading import Thread, Timer
from typing import Optional, Union, Dict

import requests
import telebot
from OpenSSL.crypto import X509
from requests.exceptions import HTTPError

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.paths import get_settings_path
from deeppavlov.utils.connector.conversation import BaseConversation, MSConversation, AlexaConversation, AliceConversation, TgConversation
from deeppavlov.utils.connector.ssl_tools import verify_cert, verify_signature

TELEGRAM_MODELS_INFO_FILENAME = 'models_info.json'

log = getLogger(__name__)

ValidatedCert = namedtuple('ValidatedCert', ['cert', 'expiration_timestamp'])


class BaseBot(Thread):
    _config: dict
    input_queue: Queue
    _run_flag: bool
    _conversations: Dict[str, BaseConversation]

    def __init__(self, model_config: Union[str, Path, dict],
                 config: dict,
                 input_queue: Queue) -> None:
        super(BaseBot, self).__init__()
        self._config = config
        self.input_queue = input_queue
        self._run_flag = True
        self._model = build_model(model_config)

        model_name = type(self._model.get_main_component()).__name__
        models_info_path = Path(get_settings_path(), TELEGRAM_MODELS_INFO_FILENAME).resolve()
        models_info = read_json(str(models_info_path))
        model_info = models_info.get(model_name, models_info['@default'])
        self._config.update(model_info)

        self._conversations = dict()
        log.info('Bot initiated')

    def run(self) -> None:
        """Thread run method implementation."""
        while self._run_flag:
            try:
                request = self.input_queue.get(timeout=1)
            except Empty:
                pass
            else:
                response = self._handle_request(request)
                self._send_response(response)

    def join(self, timeout=None):
        """Thread join method implementation."""
        self._run_flag = False
        for timer in threading.enumerate():
            if isinstance(timer, Timer):
                timer.cancel()
        Thread.join(self, timeout)

    def _del_conversation(self, conversation_key: str) -> None:
        """Deletes Conversation instance.

        Args:
            conversation_key: Conversation key.
        """
        if conversation_key in self._conversations.keys():
            del self._conversations[conversation_key]
            log.info(f'Deleted conversation, key: {conversation_key}')

    def _handle_request(self, request: dict) -> Optional[dict]:
        raise NotImplementedError

    def _send_response(self, response: Optional[dict]) -> None:
        raise NotImplementedError


class AlexaBot(BaseBot):
    """Contains agent, conversations, validates Alexa requests and routes them to conversations.

    Args:
        agent_generator: Callback which generates DefaultAgent instance with alexa skill.
        config: Alexa skill configuration settings.
        input_queue: Queue for incoming requests from Alexa.
        output_queue: Queue for outcoming responses to Alexa.

    Attributes:
        config: Alexa skill configuration settings.
        conversations: Dict with current conversations, key - Alexa user ID, value - Conversation object.
        input_queue: Queue for incoming requests from Alexa.
        output_queue: Queue for outcoming responses to Alexa.
        _valid_certificates: Dict where key - signature chain url, value - ValidatedCert instance.
        agent: Alexa skill agent.
        agent_generator: Callback which generates DefaultAgent instance with alexa skill.
        _timer: Timer which triggers periodical certificates with expired validation cleanup.
    """
    def __init__(self,
                 model_config: Union[str, Path, dict],
                 config: dict,
                 input_queue: Queue,
                 output_queue: Queue) -> None:
        super(AlexaBot, self).__init__(model_config, config, input_queue)
        self.output_queue = output_queue
        self._amazon_cert_lifetime = config['amazon_cert_lifetime']
        self._request_timestamp_tolerance_secs = config['request_timestamp_tolerance_secs']
        self._refresh_valid_certs_period_secs = config['refresh_valid_certs_period_secs']
        self._valid_certificates: Dict[str, ValidatedCert] = {}

        self._refresh_valid_certs()

    def _refresh_valid_certs(self) -> None:
        """Conducts cleanup of periodical certificates with expired validation."""
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
        """Conducts series of Alexa request verifications against Amazon Alexa requirements.

        Args:
            signature_chain_url: Signature certificate URL from SignatureCertChainUrl HTTP header.
            signature: Base64 decoded Alexa request signature from Signature HTTP header.
            request_body: full HTTPS request body
        Returns:
            result: True if verification was successful, False if not.
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
        """Processes Alexa requests from skill server and returns responses to Alexa.

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
                AlexaConversation(config=self._config,
                                  model=self._model,
                                  self_destruct_callback=lambda: self._del_conversation(conversation_key))

            log.info(f'Created new conversation, key: {conversation_key}')

        conversation = self._conversations[conversation_key]
        response = conversation.handle_request(alexa_request)

        return response

    def _send_response(self, response: dict) -> None:
        self.output_queue.put(response)


class AliceBot(BaseBot):
    def __init__(self,
                 model_config: Union[str, Path, dict],
                 config: dict,
                 input_queue: Queue,
                 output_queue: Queue) -> None:
        super(AliceBot, self).__init__(model_config, config, input_queue)
        self.output_queue = output_queue

    def _handle_request(self, request: dict) -> Optional[dict]:
        conversation_key = request['session']['session_id']

        if conversation_key not in self._conversations:
            self._conversations[conversation_key] = \
                AliceConversation(config=self._config,
                                  model=self._model,
                                  self_destruct_callback=lambda: self._del_conversation(conversation_key))
            log.info(f'Created new conversation, key: {conversation_key}')
        conversation = self._conversations[conversation_key]
        response = conversation.handle_request(request)

        return response

    def _send_response(self, response: Optional[dict]) -> None:
        self.output_queue.put(response)


class MSBot(BaseBot):
    def __init__(self,
                 model_config: Union[str, Path, dict],
                 config: dict,
                 input_queue: Queue):
        super(MSBot, self).__init__(model_config, config, input_queue)
        self._auth_polling_interval = config['auth_polling_interval']
        self._auth_url = config['auth_url']
        self._auth_headers = config['auth_headers']
        self._auth_payload = config['auth_payload']
        self._http_session = requests.Session()
        self._update_access_info()

    def _update_access_info(self):
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

    def _handle_request(self, request: dict):
        conversation_key = request['conversation']['id']

        if conversation_key not in self._conversations:
            self._conversations[conversation_key] = \
                MSConversation(config=self._config,
                               model=self._model,
                               activity=request,
                               self_destruct_callback=lambda: self._del_conversation(conversation_key),
                               http_session=self._http_session)

            log.info(f'Created new conversation, key: {conversation_key}')

        conversation = self._conversations[conversation_key]
        conversation.handle_request(request)

    def _send_response(self, response: dict) -> None:
        pass


class TelegramBot(BaseBot):
    def __init__(self, model_config: Union[str, Path, dict], config: dict):
        super(TelegramBot, self).__init__(model_config, config, Queue())
        self._token = config['token']
        self._start_message = self._config['start_message']
        self._help_message = self._config['help_message']

    def start(self):
        bot = telebot.TeleBot(self._token)
        bot.remove_webhook()

        @bot.message_handler(commands=['start'])
        def send_start_message(message):
            chat_id = message.chat.id
            out_message = self._start_message
            bot.send_message(chat_id, out_message)

        @bot.message_handler(commands=['help'])
        def send_help_message(message):
            chat_id = message.chat.id
            out_message = self._help_message
            bot.send_message(chat_id, out_message)

        @bot.message_handler()
        def handle_inference(message):
            chat_id = message.chat.id
            context = message.text

            if chat_id not in self._conversations:
                self._conversations[chat_id] = \
                    TgConversation(self._config, self._model, self._del_conversation(chat_id))

            conversation = self._conversations[chat_id]
            response = conversation.handle_request(context)
            bot.send_message(chat_id, response)

        bot.polling()

    def _handle_request(self, request: dict) -> Optional[dict]:
        pass

    def _send_response(self, response: Optional[dict]) -> None:
        pass
