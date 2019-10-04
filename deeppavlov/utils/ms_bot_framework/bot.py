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
from logging import getLogger
from pathlib import Path
from queue import Queue
from typing import Union

import requests
from requests.exceptions import HTTPError

from deeppavlov.utils.bot import BaseBot
from deeppavlov.utils.ms_bot_framework.conversation import MSConversation

log = getLogger(__name__)

ConvKey = namedtuple('ConvKey', ['channel_id', 'conversation_id'])


class MSBot(BaseBot):
    def __init__(self, model_config: Union[str, Path, dict],
                 default_skill_wrap: bool,
                 config: dict,
                 input_queue: Queue):
        super(MSBot, self).__init__(model_config, default_skill_wrap, config, input_queue)
        self.conversations = {}
        self.http_session = requests.Session()
        self._update_access_info()

    def _del_conversation(self, conversation_key: ConvKey):
        if conversation_key in self.conversations.keys():
            del self.conversations[conversation_key]
            log.info(f'Deleted conversation, key: {str(conversation_key)}')

    def _update_access_info(self):
        polling_interval = self._config['auth_polling_interval']
        self._timer = threading.Timer(polling_interval, self._update_access_info)
        self._timer.start()

        result = requests.post(url=self._config['auth_url'],
                               headers=self._config['auth_headers'],
                               data=self._config['auth_payload'])

        status_code = result.status_code
        if status_code != 200:
            raise HTTPError(f'Authentication token request returned wrong HTTP status code: {status_code}')

        access_info = result.json()
        headers = {
            'Authorization': f"{access_info['token_type']} {access_info['access_token']}",
            'Content-Type': 'application/json'
        }

        self.http_session.headers.update(headers)

        log.info(f'Obtained authentication information from Microsoft Bot Framework: {str(access_info)}')

    def _handle_request(self, activity: dict):
        conversation_key = ConvKey(activity['channelId'], activity['conversation']['id'])

        if conversation_key not in self.conversations.keys():
            self.conversations[conversation_key] = MSConversation(config=self._config,
                                                                  agent=self._agent,
                                                                  activity=activity,
                                                                  conversation_key=conversation_key,
                                                                  self_destruct_callback=lambda: self._del_conversation(conversation_key),
                                                                  http_session=self.http_session)

            log.info(f'Created new conversation, key: {str(conversation_key)}')

        conversation = self.conversations[conversation_key]
        conversation.handle_activity(activity)

    def _send_response(self, response: dict) -> None:
        pass
