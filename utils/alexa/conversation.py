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

from deeppavlov.agents.default_agent.default_agent import DefaultAgent
from deeppavlov.core.agent.rich_content import RichMessage

log = getLogger(__name__)


class Conversation:
    """Contains agent (if multi-instanced), receives requests, generates responses.

    Args:
        config: Alexa skill configuration settings.
        agent: DeepPavlov Agent instance.
        conversation_key: Alexa conversation ID.
        self_destruct_callback: Conversation instance deletion callback function.

    Attributes:
        config: Alexa skill configuration settings.
        agent: Alexa skill agent.
        key: Alexa conversation ID.
        stateful: Stateful mode flag.
        timer: Conversation self-destruct timer.
        handled_requests: Mapping of Alexa requests types to requests handlers.
        response_template: Alexa response template.
        """
    def __init__(self, config: dict, agent: DefaultAgent, conversation_key: str,
                 self_destruct_callback: callable) -> None:
        self.config = config
        self.agent = agent
        self.key = conversation_key
        self.self_destruct_callback = self_destruct_callback
        self.stateful: bool = self.config['stateful']
        self.timer: Optional[Timer] = None

        self.handled_requests = {
            'LaunchRequest': self._handle_launch,
            'IntentRequest': self._handle_intent,
            'SessionEndedRequest': self._handle_end,
            '_unsupported': self._handle_unsupported
        }

        self.response_template = {
            'version': '1.0',
            'sessionAttributes': {
                'sessionId': None
            }
        }

        self._start_timer()

    def _start_timer(self) -> None:
        """Initiates self-destruct timer."""
        self.timer = Timer(self.config['conversation_lifetime'], self.self_destruct_callback)
        self.timer.start()

    def _rearm_self_destruct(self) -> None:
        """Rearms self-destruct timer."""
        self.timer.cancel()
        self._start_timer()

    def handle_request(self, request: dict) -> dict:
        """Routes Alexa requests to appropriate handlers.

        Args:
            request: Alexa request.
        Returns:
            response: Response conforming Alexa response specification.
        """
        request_type = request['request']['type']
        request_id = request['request']['requestId']
        log.debug(f'Received request. Type: {request_type}, id: {request_id}')

        if request_type in self.handled_requests.keys():
            response: dict = self.handled_requests[request_type](request)
        else:
            response: dict = self.handled_requests['_unsupported'](request)
            log.warning(f'Unsupported request type: {request_type}, request id: {request_id}')

        self._rearm_self_destruct()

        return response

    def _act(self, utterance: str) -> list:
        """Infers DeepPavlov agent with raw user input extracted from Alexa request.

        Args:
            utterance: Raw user input extracted from Alexa request.
        Returns:
            response: DeepPavlov agent response.
        """
        if self.stateful:
            utterance = [[utterance], [self.key]]
        else:
            utterance = [[utterance]]

        agent_response: list = self.agent(*utterance)

        return agent_response

    def _generate_response(self, response: dict, request: dict) -> dict:
        """Populates generated response with additional data conforming Alexa response specification.

        Args:
            response: Raw user input extracted from Alexa request.
            request: Alexa request.
        Returns:
            response: Response conforming Alexa response specification.
        """
        response_template = deepcopy(self.response_template)
        response_template['sessionAttributes']['sessionId'] = request['session']['sessionId']

        for key, value in response_template.items():
            if key not in response.keys():
                response[key] = value

        return response

    def _handle_intent(self, request: dict) -> dict:
        """Handles IntentRequest Alexa request.

        Args:
            request: Alexa request.
        Returns:
            response: "response" part of response dict conforming Alexa specification.
        """
        intent_name = self.config['intent_name']
        slot_name = self.config['slot_name']

        request_id = request['request']['requestId']
        request_intent: dict = request['request']['intent']

        if intent_name != request_intent['name']:
            log.error(f"Wrong intent name received: {request_intent['name']} in request {request_id}")
            return {'error': 'wrong intent name'}

        if slot_name not in request_intent['slots'].keys():
            log.error(f'No slot named {slot_name} found in request {request_id}')
            return {'error': 'no slot found'}

        utterance = request_intent['slots'][slot_name]['value']
        agent_response = self._act(utterance)

        if not agent_response:
            log.error(f'Some error during response generation for request {request_id}')
            return {'error': 'error during response generation'}

        prediction: RichMessage = agent_response[0]
        prediction: list = prediction.alexa()

        if not prediction:
            log.error(f'Some error during response generation for request {request_id}')
            return {'error': 'error during response generation'}

        response = self._generate_response(prediction[0], request)

        return response

    def _handle_launch(self, request: dict) -> dict:
        """Handles LaunchRequest Alexa request.

        Args:
            request: Alexa request.
        Returns:
            response: "response" part of response dict conforming Alexa specification.
        """
        response = {
            'response': {
                'shouldEndSession': False,
                'outputSpeech': {
                    'type': 'PlainText',
                    'text': self.config['start_message']
                },
                'card': {
                    'type': 'Simple',
                    'content': self.config['start_message']
                }
            }
        }

        response = self._generate_response(response, request)

        return response

    def _handle_end(self, request: dict) -> dict:
        """Handles SessionEndedRequest Alexa request and deletes Conversation instance.

        Args:
            request: Alexa request.
        Returns:
            response: Dummy empty response dict.
        """
        response = {}
        self.self_destruct_callback()
        return response

    def _handle_unsupported(self, request: dict) -> dict:
        """Handles all unsupported types of Alexa requests. Returns standard message.

        Args:
            request: Alexa request.
        Returns:
            response: "response" part of response dict conforming Alexa specification.
        """
        response = {
            'response': {
                'shouldEndSession': False,
                'outputSpeech': {
                    'type': 'PlainText',
                    'text': self.config['unsupported_message']
                },
                'card': {
                    'type': 'Simple',
                    'content': self.config['unsupported_message']
                }
            }
        }

        response = self._generate_response(response, request)

        return response
