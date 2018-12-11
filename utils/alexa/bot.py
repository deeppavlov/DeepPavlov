import threading
from datetime import timedelta, datetime
from queue import Queue
from threading import Thread

from utils.alexa.conversation import Conversation
from deeppavlov.core.common.log import get_logger
from utils.alexa.ssl_tools import verify_cert

REQUEST_TIMESTAMP_TOLERANCE_SECS = 150

log = get_logger(__name__)


class Bot(Thread):
    def __init__(self, agent_generator: callable, config: dict, input_queue: Queue, output_queue: Queue):
        super(Bot, self).__init__()
        self.config = config

        self.conversations = {}
        self.valid_certificates = {}
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.agent = None
        self.agent_generator = agent_generator

        if not self.config['multi_instance']:
            self.agent = self._init_agent()
            log.info('New bot instance level agent initiated')

        amazon_cert_lifetime: timedelta = self.config['amazon_cert_lifetime']
        amazon_cert_lifetime: int = int(amazon_cert_lifetime.total_seconds())

        self.timer = threading.Timer(amazon_cert_lifetime, self._refresh_valid_certs)
        self.timer.start()

    def run(self):
        while True:
            request = self.input_queue.get()
            response = self._handle_request(request)
            self.output_queue.put(response)

    def del_conversation(self, conversation_key: str):
        del self.conversations[conversation_key]
        log.info(f'Deleted conversation, key: {conversation_key}')

    def _init_agent(self):
        # TODO: Decide about multi-instance mode necessity.
        # If model multi-instancing is still necessary - refactor and remove
        agent = self.agent_generator()
        return agent

    def _refresh_valid_certs(self):
        amazon_cert_lifetime: timedelta = self.config['amazon_cert_lifetime']
        amazon_cert_lifetime: int = int(amazon_cert_lifetime.total_seconds())

        self.timer = threading.Timer(amazon_cert_lifetime, self._refresh_valid_certs)
        self.timer.start()

        for valid_cert_url, cert_expiration_time in self.valid_certificates.items():
            cert_expiration_time: datetime = cert_expiration_time
            if datetime.utcnow() > cert_expiration_time:
                del self.valid_certificates[valid_cert_url]
                log.info(f'Validation period of {valid_cert_url} certificate expired')

    def _handle_request(self, request: dict) -> dict:
        request_body: bytes = request['request_body']
        signature_chain_url: str = request['signature_chain_url']
        signature: str = request['signature']
        alexa_request: dict = request['alexa_request']

        if not verify_cert(signature_chain_url, signature, request_body):
            return {'error': 'failed certificate/signature check'}

        timestamp_str = alexa_request['request']['timestamp']
        timestamp_datetime = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
        now = datetime.utcnow()
        delta = now - timestamp_datetime

        if abs(delta.seconds) > REQUEST_TIMESTAMP_TOLERANCE_SECS:
            return {'error': 'failed request timestamp check'}

        # request_type = alexa_request['request']['type']
        conversation_key = alexa_request['session']['user']['userId']

        if conversation_key not in self.conversations.keys():
            if self.config['multi_instance']:
                conv_agent = self._init_agent()
                log.info('New conversation instance level agent initiated')
            else:
                conv_agent = self.agent

            self.conversations[conversation_key] = Conversation(bot=self,
                                                                agent=conv_agent,
                                                                conversation_key=conversation_key)

            log.info(f'Created new conversation, key: {conversation_key}')

        conversation = self.conversations[conversation_key]
        response = conversation.handle_request(alexa_request)

        return response
