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
    'AlexaConversation': 'alexa',
    'AliceConversation': 'alice',
    'MSConversation': 'ms_bot_framework',
    'TelegramConversation': 'telegram',
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


class AlexaConversation(BaseConversation):
    """Receives requests from Amazon Alexa and generates responses."""
    _intent_name: str
    _slot_name: str
    _handled_requests: Dict[str, callable]

    def __init__(self, config: dict, model, self_destruct_callback: callable, conversation_id: str) -> None:
        super(AlexaConversation, self).__init__(config, model, self_destruct_callback, conversation_id)
        self._intent_name = config['intent_name']
        self._slot_name = config['slot_name']

        self._handled_requests = {
            'LaunchRequest': self._handle_launch,
            'IntentRequest': self._handle_intent,
            'SessionEndedRequest': self._handle_end,
            '_unsupported': self._handle_unsupported
        }

    def _handle_request(self, request: dict) -> dict:
        """Routes Alexa requests to the appropriate handler.

        Args:
            request: Alexa request.

        Returns:
            response: Response conforming to the Alexa response specification.

        """
        request_type = request['request']['type']
        request_id = request['request']['requestId']
        log.debug(f'Received request. Type: {request_type}, id: {request_id}')

        if request_type in self._handled_requests:
            response = self._handled_requests[request_type](request)
        else:
            response = self._handled_requests['_unsupported'](request)

        return response

    def _generate_response(self, message: str, request: dict) -> dict:
        """Wraps message in the conforming to the Alexa data structure.

        Args:
            message: Raw message to be sent to Alexa.
            request: Request from the channel to which the ``message`` replies.

        Returns:
            response: Data structure conforming to the Alexa response specification.

        """
        response = {
            'version': '1.0',
            'sessionAttributes': {
                'sessionId': request['session']['sessionId']
            },
            'response': {
                'shouldEndSession': False,
                'outputSpeech': {
                    'type': 'PlainText',
                    'text': message
                },
                'card': {
                    'type': 'Simple',
                    'content': message
                }
            }
        }

        return response

    def _handle_intent(self, request: dict) -> dict:
        """Handles IntentRequest Alexa request.

        Args:
            request: Alexa request.

        Returns:
            response: Data structure conforming to the Alexa response specification.

        """
        request_id = request['request']['requestId']
        request_intent: dict = request['request']['intent']

        if self._intent_name != request_intent['name']:
            log.error(f"Wrong intent name received: {request_intent['name']} in request {request_id}")
            return {'error': 'wrong intent name'}

        if self._slot_name not in request_intent['slots'].keys():
            log.error(f'No slot named {self._slot_name} found in request {request_id}')
            return {'error': 'no slot found'}

        utterance = request_intent['slots'][self._slot_name]['value']
        model_response = self._act(utterance)

        if not model_response:
            log.error(f'Some error during response generation for request {request_id}')
            return {'error': 'error during response generation'}

        response = self._generate_response(model_response, request)

        return response

    def _handle_end(self, request: dict) -> dict:
        """Handles SessionEndedRequest Alexa request and deletes Conversation instance.

        Args:
            request: Alexa request.

        Returns:
            response: Dummy empty response dict.

        """
        response = {}
        self._self_destruct_callback(self._conversation_id)
        return response


class AliceConversation(BaseConversation):
    """Receives requests from Yandex.Alice and generates responses."""
    def _handle_request(self, request: dict) -> dict:
        """Routes Alice requests to the appropriate handler.

        Args:
            request: Alice request.

        Returns:
            response: Response conforming to the Alice response specification.

        """
        message_id = request['session']['message_id']
        session_id = request['session']['session_id']
        log.debug(f'Received message. Session: {session_id}, message_id: {message_id}')

        if request['session']['new']:
            response = self._handle_launch(request)
        elif request['request']['command'].strip():
            text = request['request']['command'].strip()
            model_response = self._act(text)
            response = self._generate_response(model_response, request)
        else:
            response = self._handle_unsupported(request)

        return response

    def _generate_response(self, message: str, request: dict) -> dict:
        """Wraps message in the conforming to the Alice data structure.

        Args:
            message: Raw message to be sent to Alice.
            request: Request from the channel to which the ``message`` replies.

        Returns:
            response: Data structure conforming to the Alice response specification.

        """
        response = {
            'response': {
                'end_session': False,
                'text': message
            },
            'session': {
                'session_id': request['session']['session_id'],
                'message_id': request['session']['message_id'],
                'user_id': request['session']['user_id']
            },
            'version': '1.0'
        }

        return response


class MSConversation(BaseConversation):
    """Receives requests from Microsoft Bot Framework and generates responses."""
    def __init__(self,
                 config: dict,
                 model: Chainer,
                 self_destruct_callback: callable,
                 conversation_id: str,
                 http_session: Session) -> None:
        """Initiates instance properties and starts self-destruct timer.

        Args:
            config: Dictionary containing base conversation parameters.
            model: Model that infered with user messages.
            self_destruct_callback: Function that removes this Conversation instance.
            conversation_id: Conversation ID.
            http_session: Session used to send responses to Bot Framework.

        """
        super(MSConversation, self).__init__(config, model, self_destruct_callback, conversation_id)
        self._http_session = http_session

        self._handled_activities = {
            'message': self._handle_message,
            'conversationUpdate': self._handle_launch,
            '_unsupported': self._handle_unsupported
        }

    def _handle_request(self, request: dict) -> None:
        """Routes MS Bot requests to the appropriate handler. Returns None since handlers send responses themselves.

        Args:
            request: MS Bot request.

        """
        activity_type = request['type']
        activity_id = request['id']
        log.debug(f'Received activity. Type: {activity_type}, id: {activity_id}')

        if activity_type in self._handled_activities.keys():
            self._handled_activities[activity_type](request)
        else:
            self._handled_activities['_unsupported'](request)

        self._rearm_self_destruct()

    def _handle_message(self, request: dict) -> None:
        """Handles MS Bot message request.

        Request redirected to ``_unsupported`` handler if ms bot message does not contain raw text.

        Args:
            request: MS Bot request.

        """
        if 'text' in request:
            in_text = request['text']
            model_response = self._act(in_text)
            if model_response:
                self._generate_response(model_response, request)
        else:
            self._handled_activities['_unsupported'](request)

    def _generate_response(self, message: str, request: dict) -> None:
        """Wraps message in the conforming to the MS Bot data structure and sends it to MS Bot via HTTP session.

        Args:
            message: Raw message to be sent to MS Bot.
            request: Request from the channel to which the ``message`` replies.

        """
        response = {
            "type": "message",
            "from": request['recipient'],
            "recipient": request['from'],
            'conversation': request['conversation'],
            'text': message
        }

        url = urljoin(request['serviceUrl'], f"v3/conversations/{request['conversation']['id']}/activities")

        response = self._http_session.post(url=url, json=response)

        try:
            response_json_str = str(response.json())
        except ValueError as e:
            response_json_str = repr(e)

        log.debug(f'Sent activity to the MSBotFramework server. '
                  f'Response code: {response.status_code}, response contents: {response_json_str}')


class TelegramConversation(BaseConversation):
    """Receives requests from Telegram bot and generates responses."""
    def _handle_request(self, message: str) -> str:
        """Handles raw text message from Telegram bot.

        Args:
            message: Message from Telegram bot.

        Returns:
            response: Response to a ``message``.

        """
        response = self._act(message)

        return response

    def _generate_response(self, message: str, request: dict) -> None:
        """Does nothing."""
        pass
