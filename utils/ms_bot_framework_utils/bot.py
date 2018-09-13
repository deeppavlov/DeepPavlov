import threading
from queue import Queue
from threading import Thread
from collections import namedtuple

import requests
from requests.exceptions import HTTPError

from .model import Model
from .conversation import Conversation
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)

ConvKey = namedtuple('ConvKey', ['channel_id', 'conversation_id'])


class Bot(Thread):
    def __init__(self, config: dict, input_queue: Queue):
        super(Bot, self).__init__()
        self.config = config

        self.conversations = {}
        self.access_info = {}
        self.http_sessions = {}
        self.input_queue = input_queue

        self.model = None
        if not self.config['multi_instance']:
            self.model = self._init_model(self.config)
            log.info('New model bot instance level model initiated')

        polling_interval = self.config['auth_polling_interval']
        self.timer = threading.Timer(polling_interval, self._update_access_info)
        self._request_access_info()
        self.timer.start()

    def run(self):
        while True:
            activity = self.input_queue.get()
            self._handle_activity(activity)

    def del_conversation(self, conversation_key: ConvKey):
        del self.conversations[conversation_key]
        log.info(f'Deleted conversation, key: {str(conversation_key)}')

    def _init_model(self, server_config: dict):
        model = Model(server_config)
        return model

    def _update_access_info(self):
        polling_interval = self.config['auth_polling_interval']
        self.timer = threading.Timer(polling_interval, self._update_access_info)
        self.timer.start()
        self._request_access_info()

    def _request_access_info(self):
        headers = {'Host': self.config['auth_host'],
                   'Content-Type': self.config['auth_content_type']}

        payload = {'grant_type': self.config['auth_grant_type'],
                   'scope': self.config['auth_scope'],
                   'client_id': self.config['auth_app_id'],
                   'client_secret': self.config['auth_app_secret']}

        result = requests.post(url=self.config['auth_url'],
                               headers=headers,
                               data=payload)

        status_code = result.status_code
        if status_code != 200:
            raise HTTPError(f'Authentication token request returned wrong HTTP status code: {status_code}')

        self.access_info = result.json()
        log.info(f'Obtained authentication information from Microsoft Bot Framework: {str(self.access_info)}')

    def _handle_activity(self, activity: dict):
        conversation_key = ConvKey(activity['channelId'], activity['conversation']['id'])

        if conversation_key not in self.conversations.keys():
            if self.config['multi_instance']:
                conv_model = self._init_model(self.config)
                log.info('New model conversation instance level model initiated')
            else:
                conv_model = self.model

            conversation_lifetime = self.config['conversation_lifetime']

            self.conversations[conversation_key] = Conversation(bot=self,
                                                                model=conv_model,
                                                                activity=activity,
                                                                conversation_key=conversation_key,
                                                                conversation_lifetime=conversation_lifetime)

            log.info(f'Created new conversation, key: {str(conversation_key)}')

        conversation = self.conversations[conversation_key]
        conversation.handle_activity(activity)
