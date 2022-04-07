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

from logging import getLogger
from threading import Timer
from typing import Dict, Optional, Union
from urllib.parse import urljoin

from requests import Session

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.utils.connector.dialog_logger import DialogLogger

log = getLogger(__name__)

DIALOG_LOGGER_NAME_MAPPING = {
    '_unsupported': 'new_conversation'
}


class BaseConversation:
    """Receives requests, generates responses."""
    _model: Chainer
    _self_destruct_callback: callable
    _conversation_id: Union[int, str]
    _timer: Timer
    _infer_utterances: list
    _conversation_lifetime: int
    _next_arg_msg: str
    _start_message: str

    def __init__(self,
                 config: dict,
                 model: Chainer,
                 self_destruct_callback: callable,
                 conversation_id: Union[int, str]) -> None:
        """Initiates instance properties and starts self-destruct timer.

        Args:
            config: Dictionary containing base conversation parameters.
            model: Model that infered with user messages.
            self_destruct_callback: Function that removes this Conversation instance.
            conversation_id: Conversation ID.

        """
        self._model = model
        self._self_destruct_callback = self_destruct_callback
        self._conversation_id = conversation_id
        self._infer_utterances = list()
        self._conversation_lifetime = config['conversation_lifetime']
        self._next_arg_msg = config['next_argument_message']
        self._start_message = config['start_message']
        self._unsupported_message = config['unsupported_message']
        logger_name: str = DIALOG_LOGGER_NAME_MAPPING.get(type(self).__name__,
                                                          DIALOG_LOGGER_NAME_MAPPING['_unsupported'])
        self._dialog_logger = DialogLogger(logger_name=logger_name)
        self._start_timer()

    def handle_request(self, request: dict) -> Optional[dict]:
        """Rearms self-destruct timer and sends the request to a processing.

        Args:
            request: Request from the channel.

        Returns:
            response: Corresponding to the channel response to the request from the channel if replies are sent via bot,
                None otherwise.

        """
        self._rearm_self_destruct()
        return self._handle_request(request)

    def _start_timer(self) -> None:
        """Initiates self-destruct timer."""
        self._timer = Timer(self._conversation_lifetime, self._self_destruct_callback, [self._conversation_id])
        self._timer.start()

    def _rearm_self_destruct(self) -> None:
        """Rearms self-destruct timer."""
        self._timer.cancel()
        self._start_timer()

    def _handle_request(self, request: dict) -> Optional[dict]:
        """Routes the request to the appropriate handler.

        Args:
            request: Request from the channel.

        Returns:
            response: Corresponding response to the channel request if replies are sent via bot, None otherwise.

        """
        raise NotImplementedError

    def _handle_launch(self, request: dict) -> Optional[dict]:
        """Handles launch request.

        Args:
            request: Start request from channel.

        Returns:
            response: Greeting message wrapped in the appropriate to the channel structure if replies are sent via bot,
                None otherwise.

        """
        response = self._generate_response(self._start_message, request)

        return response

    def _handle_unsupported(self, request: dict) -> Optional[dict]:
        """Handles all unsupported request types.

        Args:
            request: Request from channel for which a separate handler was not defined.

        Returns:
            response: Message that request type is not supported wrapped in the appropriate to the channel data
                structure if replies are sent via bot, None otherwise.

        """
        response = self._generate_response(self._unsupported_message, request)
        log.warning(f'Unsupported request: {request}')

        return response

    def _generate_response(self, message: str, request: dict) -> Optional[dict]:
        """Wraps message in the appropriate to the channel data structure.

        Args:
            message: Raw message to be sent to the channel.
            request: Request from the channel to which the ``message`` replies.

        Returns:
            response: Data structure to be sent to the channel if replies are sent via bot, None otherwise.

        """
        raise NotImplementedError

    def _act(self, utterance: str) -> str:
        """Infers DeepPavlov model with utterance.

        If DeepPavlov model requires more than one argument, utterances are accumulated until reaching required
        arguments amount to infer.

        Args:
            utterance: Text to be processed by DeepPavlov model.

        Returns:
            response: Model response if enough model arguments have been accumulated, message prompting for the next
                model argument otherwise.

        """
        self._infer_utterances.append([utterance])

        if len(self._infer_utterances) == len(self._model.in_x):
            self._dialog_logger.log_in(self._infer_utterances, self._conversation_id)
            prediction = self._model(*self._infer_utterances)
            self._infer_utterances = list()
            if len(self._model.out_params) == 1:
                prediction = [prediction]
            prediction = '; '.join([str(output[0]) for output in prediction])
            response = prediction
            self._dialog_logger.log_out(response, self._conversation_id)
        else:
            response = self._next_arg_msg.format(self._model.in_x[len(self._infer_utterances)])

        return response
