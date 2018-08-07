import threading
import requests
from multiprocessing import Process, Queue
from requests.exceptions import HTTPError

from conversation import Conversation
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model_from_config

log = get_logger(__name__)


class Bot(Process):
    def __init__(self, config: dict, model_config_path: str, input_queue: Queue):
        super(Bot, self).__init__()
        self.config = config
        self.model = self._init_model(model_config_path)
        self.conversations = {}
        self.access_info = {}
        self.http_sessions = {}
        self.input_queue = input_queue

        self._request_access_info()
        polling_interval = self.config['ms_bot_framework_defaults']['auth_polling_interval']
        timer = threading.Timer(polling_interval, self._update_access_info)
        timer.start()

    def run(self):
        while True:
            activity = self.input_queue.get()
            self._handle_activity(activity)

    def _init_model(self, model_config_path):
        model_config = read_json(model_config_path)
        model = build_model_from_config(model_config)
        return model

    def _update_access_info(self):
        polling_interval = self.config['ms_bot_framework_defaults']['auth_polling_interval']
        timer = threading.Timer(polling_interval, self._update_access_info)
        timer.start()
        self._request_access_info()

    def _request_access_info(self):
        headers = {'Host': self.config['ms_bot_framework_defaults']['auth_host'],
                   'Content-Type': self.config['ms_bot_framework_defaults']['auth_content_type']}

        payload = {'grant_type': self.config['ms_bot_framework_defaults']['auth_grant_type'],
                   'scope': self.config['ms_bot_framework_defaults']['auth_scope'],
                   'client_id': self.config['ms_bot_framework_defaults']['auth_client_id'],
                   'client_secret': self.config['ms_bot_framework_defaults']['auth_client_secret']}

        result = requests.post(url=self.config['ms_bot_framework_defaults']['auth_url'],
                               headers=headers,
                               data=payload)

        # TODO: insert json content to the error message
        status_code = result.status_code
        if status_code != 200:
            raise HTTPError(f'Authentication token request returned wrong HTTP status code: {status_code}')

        self.access_info = result.json()
        log.info(f'Obtained authentication information from Microsoft Bot Framework: {str(self.access_info)}')

    def _handle_activity(self, activity: dict):
        conversation_key = f"{activity['channelId']}||{activity['conversation']['id']}"

        if conversation_key not in self.conversations.keys():
            self.conversations[conversation_key] = Conversation(self, activity)
            log.info(f'Created new conversation {conversation_key}')

        conversation = self.conversations[conversation_key]
        conversation.handle_activity(activity)
