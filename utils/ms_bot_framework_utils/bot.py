import threading
from queue import Queue
from threading import Thread
from collections import namedtuple

import requests
from requests.exceptions import HTTPError

from .conversation import Conversation
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.agent.agent import Agent
from deeppavlov.agents.default_agent.default_processors import DefaultRichContentWrapper
from deeppavlov.skills.default_skill.default_skill import DefaultStatelessSkill

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

        self.agent = None
        if not self.config['multi_instance']:
            self.agent = self._init_agent()
            log.info('New bot instance level agent initiated')

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

    def _init_agent(self):
        model_config = read_json(self.config['model_config_path'])
        model = build_model_from_config(model_config)
        skill = DefaultStatelessSkill(model)
        agent = Agent([skill], skills_processor=DefaultRichContentWrapper())
        return agent

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
                conv_agent = self._init_agent()
                log.info('New conversation instance level agent initiated')
            else:
                conv_agent = self.agent

            self.conversations[conversation_key] = Conversation(bot=self,
                                                                agent=conv_agent,
                                                                activity=activity,
                                                                conversation_key=conversation_key)

            log.info(f'Created new conversation, key: {str(conversation_key)}')

        conversation = self.conversations[conversation_key]
        conversation.handle_activity(activity)
