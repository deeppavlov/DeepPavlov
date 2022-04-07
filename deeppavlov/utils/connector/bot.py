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
from deeppavlov.utils.connector.conversation import BaseConversation
from deeppavlov.utils.connector.conversation import MSConversation
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
