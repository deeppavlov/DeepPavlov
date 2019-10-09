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

from copy import deepcopy
from logging import getLogger
from threading import Timer
from typing import Optional
from urllib.parse import urljoin

from requests import Session

from deeppavlov.core.common.chainer import Chainer

log = getLogger(__name__)


class BaseConversation:
    _model: Chainer
    _timer: Timer
    _infer_utterances: list

    def __init__(self, config: dict, model: Chainer, self_destruct_callback: callable):
        self._model = model
        self._self_destruct_callback = self_destruct_callback
        self._infer_utterances = list()
        self._conversation_lifetime = config['conversation_lifetime']
        self._next_utter_msg = config['next_utter_msg']
        self._start_message = config['start_message']
        self._start_timer()

    def handle_request(self, request: dict) -> Optional[dict]:
        self._rearm_self_destruct()
        return self._handle_request(request)

    def _start_timer(self) -> None:
        """Initiates self-destruct timer."""
        self._timer = Timer(self._conversation_lifetime, self._self_destruct_callback)
        self._timer.start()

    def _rearm_self_destruct(self) -> None:
        """Rearms self-destruct timer."""
        self._timer.cancel()
        self._start_timer()

    def _handle_request(self, request: dict) -> Optional[dict]:
        raise NotImplementedError

    def _act(self, utterance: str) -> str:
        """Infers DeepPavlov model with raw user input extracted from request.

        Args:
            utterance: Raw user input extracted from request.

        Returns:
            response: DeepPavlov model response if  ``next_utter_msg`` from config with.

        """
        self._infer_utterances.append([utterance])
        if len(self._infer_utterances) == len(self._model.in_x):
            prediction = self._model(*self._infer_utterances)
            self._infer_utterances = list()
            if len(self._model.out_params) == 1:
                prediction = [prediction]
            prediction = '; '.join([str(output[0]) for output in prediction])
            response = prediction
        else:
            response = self._next_utter_msg.format(self._model.in_x[len(self._infer_utterances)])
        return response


class AlexaConversation(BaseConversation):
    """Contains agent, receives requests, generates responses.

    Args:
        config: Alexa skill configuration settings.
        agent: DeepPavlov Agent instance.
        conversation_key: Alexa conversation ID.
        self_destruct_callback: Conversation instance deletion callback function.

    Attributes:
        config: Alexa skill configuration settings.
        agent: Alexa skill agent.
        key: Alexa conversation ID.
        timer: Conversation self-destruct timer.
        _handled_requests: Mapping of Alexa requests types to requests handlers.
        _response_template: Alexa response template.
        """
    def __init__(self, config: dict, model, self_destruct_callback: callable) -> None:
        super(AlexaConversation, self).__init__(config, model, self_destruct_callback)
        self._intent_name = config['intent_name']
        self._slot_name = config['slot_name']

        self._handled_requests = {
            'LaunchRequest': self._handle_launch,
            'IntentRequest': self._handle_intent,
            'SessionEndedRequest': self._handle_end,
            '_unsupported': self._handle_unsupported
        }

        self._response_template = {
            'version': '1.0',
            'sessionAttributes': {
                'sessionId': None
            },
            'response': {
                'shouldEndSession': False,
                'outputSpeech': {
                    'type': 'PlainText',
                    'text': None
                },
                'card': {
                    'type': 'Simple',
                    'content': None
                }
            }
        }

    def _handle_request(self, request: dict) -> dict:
        """Routes Alexa requests to appropriate handlers.

        Args:
            request: Alexa request.
        Returns:
            response: Response conforming Alexa response specification.
        """
        request_type = request['request']['type']
        request_id = request['request']['requestId']
        log.debug(f'Received request. Type: {request_type}, id: {request_id}')

        if request_type in self._handled_requests.keys():
            response: dict = self._handled_requests[request_type](request)
        else:
            response: dict = self._handled_requests['_unsupported'](request)
            log.warning(f'Unsupported request type: {request_type}, request id: {request_id}')

        self._rearm_self_destruct()

        return response

    def _generate_response(self, text: str, request: dict) -> dict:
        """Populates generated response with additional data conforming Alexa response specification.

        Args:
            response: Raw user input extracted from Alexa request.
            request: Alexa request.
        Returns:
            response: Response conforming Alexa response specification.
        """
        response = deepcopy(self._response_template)
        response['sessionAttributes']['sessionId'] = request['session']['sessionId']

        response['response']['outputSpeech']['text'] = response['response']['card']['content'] = text

        return response

    def _handle_intent(self, request: dict) -> dict:
        """Handles IntentRequest Alexa request.

        Args:
            request: Alexa request.
        Returns:
            response: "response" part of response dict conforming Alexa specification.
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

    def _handle_launch(self, request: dict) -> dict:
        """Handles LaunchRequest Alexa request.

        Args:
            request: Alexa request.
        Returns:
            response: "response" part of response dict conforming Alexa specification.

        """
        response = self._generate_response(self._start_message, request)

        return response

    def _handle_end(self, request: dict) -> dict:
        """Handles SessionEndedRequest Alexa request and deletes Conversation instance.

        Args:
            request: Alexa request.
        Returns:
            response: Dummy empty response dict.
        """
        response = {}
        self._self_destruct_callback()
        return response

    def _handle_unsupported(self, request: dict) -> dict:
        """Handles all unsupported types of Alexa requests. Returns standard message.

        Args:
            request: Alexa request.
        Returns:
            response: "response" part of response dict conforming Alexa specification.
        """
        response = self._generate_response('Got unsupported message', request)

        return response


class AliceConversation(BaseConversation):
    def __init__(self, config, model, self_destruct_callback):
        super(AliceConversation, self).__init__(config, model, self_destruct_callback)
        self._response_template = {
            'response': {
                'end_session': False,
                'text': None
            },
            'session': {
                'session_id': None,
                'message_id': None,
                'user_id': None
            },
            'version': '1.0'
        }

    def _handle_request(self, data: dict):
        if data['session']['new']:
            response = self._generate_response(self._start_message, data)
        elif data['request']['command'].strip():
            text = data['request']['command'].strip()
            model_response = self._act(text)
            response = self._generate_response(model_response, data)
        else:
            response = self._generate_response('got unsupported message', data)
        self._rearm_self_destruct()
        return response

    def _handle_launch(self, data: dict):
        return self._generate_response(self._start_message, data)

    def _generate_response(self, text, request: dict) -> dict:
        response = deepcopy(self._response_template)

        for key in ['session_id', 'user_id', 'message_id']:
            response['session'][key] = request['session'][key]

        response['response']['text'] = text

        return response


class MSConversation(BaseConversation):
    def __init__(self, config, model, activity: dict, self_destruct_callback: callable,
                 http_session: Session) -> None:
        super(MSConversation, self).__init__(config, model, self_destruct_callback)
        self._service_url = activity['serviceUrl']
        self._conversation_id = activity['conversation']['id']

        self._http_session = http_session

        self._handled_activities = {
            'message': self._handle_message,
            'conversationUpdate': self._handle_update,
            '_unsupported': self._handle_usupported
        }

        self._response_template = {
            "type": "message",
            "from": activity['recipient'],
            "recipient": activity['from'],
            'conversation': activity['conversation'],
            'text': 'default_text'
        }

    def _handle_request(self, request: dict):
        activity_type = request['type']
        activity_id = request['id']
        log.debug(f'Received activity. Type: {activity_type}, id: {activity_id}')

        if activity_type in self._handled_activities.keys():
            self._handled_activities[activity_type](request)
        else:
            self._handled_activities['_unsupported'](request)
            log.warning(f'Unsupported activity type: {activity_type}, activity id: {activity_id}')

        self._rearm_self_destruct()

    def _handle_usupported(self, in_activity: dict) -> None:
        activity_type = in_activity['type']
        self._send_plain_text(f'Unsupported kind of {activity_type} activity!')
        log.warning(f'Received message with unsupported type: {str(in_activity)}')

    def _handle_message(self, in_activity: dict) -> None:
        if 'text' in in_activity.keys():
            in_text = in_activity['text']
            agent_response = self._act(in_text)
            if agent_response:
                self._send_plain_text(agent_response)
        else:
            self._handle_usupported(in_activity)

    def _send_plain_text(self, text: str) -> None:
        response = deepcopy(self._response_template)
        response['text'] = text

        url = urljoin(self._service_url, f"v3/conversations/{self._conversation_id}/activities")

        response = self._http_session.post(url=url, json=response)

        try:
            response_json_str = str(response.json())
        except ValueError:
            response_json_str = ''

        log.debug(f'Sent activity to the MSBotFramework server. '
                  f'Response code: {response.status_code}, response contents: {response_json_str}')

    def _handle_update(self, in_activity: dict) -> None:
        self._send_plain_text(self._start_message)


class TgConversation(BaseConversation):
    def __init__(self, config, model, self_destruct_callback):
        super(TgConversation, self).__init__(config, model, self_destruct_callback)

    def _handle_request(self, text):
        self._rearm_self_destruct()
        return self._act(text)
