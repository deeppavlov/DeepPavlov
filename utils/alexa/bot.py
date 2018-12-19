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

from threading import Timer, Thread
from datetime import timedelta, datetime
from queue import Queue
from typing import Optional, Dict
from collections import namedtuple

from OpenSSL.crypto import X509

from utils.alexa.conversation import Conversation
from utils.alexa.ssl_tools import verify_cert, verify_signature
from deeppavlov.core.common.log import get_logger
from deeppavlov.agents.default_agent.default_agent import DefaultAgent

REQUEST_TIMESTAMP_TOLERANCE_SECS = 150
REFRESH_VALID_CERTS_PERIOD_SECS = 120

log = get_logger(__name__)

ValidatedCert = namedtuple('ValidatedCert', ['cert', 'expiration_timestamp'])


class Bot(Thread):
    """Contains agent (if not multi-instanced), conversations, validates Alexa requests and routes them to conversations.

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
        valid_certificates: Dict where key - signature chain url, value - ValidatedCert instance.
        agent: Alexa skill agent if not multi-instance mode.
        agent_generator: Callback which generates DefaultAgent instance with alexa skill.
        timer: Timer which triggers periodical certificates with expired validation cleanup.
    """
    def __init__(self, agent_generator: callable, config: dict, input_queue: Queue, output_queue: Queue) -> None:
        super(Bot, self).__init__()
        self.config = config
        self.conversations: Dict[str, Conversation] = {}
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.valid_certificates: Dict[str, ValidatedCert] = {}

        self.agent: Optional[DefaultAgent] = None
        self.agent_generator = agent_generator

        if not self.config['multi_instance']:
            self.agent = self._init_agent()
            log.info('New bot instance level agent initiated')

        self.timer = Timer(REFRESH_VALID_CERTS_PERIOD_SECS, self._refresh_valid_certs)
        self.timer.start()

    def run(self) -> None:
        """Thread run method implementation."""
        while True:
            request = self.input_queue.get()
            response = self._handle_request(request)
            self.output_queue.put(response)

    def _del_conversation(self, conversation_key: str) -> None:
        """Deletes Conversation instance.

        Args:
            conversation_key: Conversation key.
        """
        if conversation_key in self.conversations.keys():
            del self.conversations[conversation_key]
            log.info(f'Deleted conversation, key: {conversation_key}')

    def _init_agent(self) -> DefaultAgent:
        """Initiates Alexa skill agent from agent generator"""
        # TODO: Decide about multi-instance mode necessity.
        # If model multi-instancing is still necessary - refactor and remove
        agent = self.agent_generator()
        return agent

    def _refresh_valid_certs(self) -> None:
        """Conducts cleanup of periodical certificates with expired validation."""
        self.timer = Timer(REFRESH_VALID_CERTS_PERIOD_SECS, self._refresh_valid_certs)
        self.timer.start()

        expired_certificates = []

        for valid_cert_url, valid_cert in self.valid_certificates.items():
            valid_cert: ValidatedCert = valid_cert
            cert_expiration_time: datetime = valid_cert.expiration_timestamp
            if datetime.utcnow() > cert_expiration_time:
                expired_certificates.append(valid_cert_url)

        for expired_cert_url in expired_certificates:
            del self.valid_certificates[expired_cert_url]
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
        if signature_chain_url not in self.valid_certificates.keys():
            amazon_cert: X509 = verify_cert(signature_chain_url)
            if amazon_cert:
                amazon_cert_lifetime: timedelta = self.config['amazon_cert_lifetime']
                expiration_timestamp = datetime.utcnow() + amazon_cert_lifetime
                validated_cert = ValidatedCert(cert=amazon_cert, expiration_timestamp=expiration_timestamp)
                self.valid_certificates[signature_chain_url] = validated_cert
                log.info(f'Certificate {signature_chain_url} validated')
            else:
                log.error(f'Certificate {signature_chain_url} validation failed')
                return False
        else:
            validated_cert: ValidatedCert = self.valid_certificates[signature_chain_url]
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

        if abs(delta.seconds) > REQUEST_TIMESTAMP_TOLERANCE_SECS:
            log.error(f'Failed timestamp check for request: {request_body.decode("utf-8", "replace")}')
            return {'error': 'failed request timestamp check'}

        conversation_key = alexa_request['session']['user']['userId']

        if conversation_key not in self.conversations.keys():
            if self.config['multi_instance']:
                conv_agent = self._init_agent()
                log.info('New conversation instance level agent initiated')
            else:
                conv_agent = self.agent

            self.conversations[conversation_key] = \
                Conversation(config=self.config,
                             agent=conv_agent,
                             conversation_key=conversation_key,
                             self_destruct_callback=lambda: self._del_conversation(conversation_key))

            log.info(f'Created new conversation, key: {conversation_key}')

        conversation = self.conversations[conversation_key]
        response = conversation.handle_request(alexa_request)

        return response
