import threading
from copy import deepcopy

from deeppavlov.core.agent.rich_content import RichMessage
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


class Conversation:
    def __init__(self, bot, agent, conversation_key: str):
        self.bot = bot
        self.agent = agent
        self.key = conversation_key

        self.stateful = self.bot.config['stateful']

        self.conversation_lifetime = self.bot.config['conversation_lifetime']
        self.timer = None
        self._start_timer()

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

    def _start_timer(self) -> None:
        self.timer = threading.Timer(self.conversation_lifetime, self._self_destruct)
        self.timer.start()

    def _rearm_self_destruct(self) -> None:
        self.timer.cancel()
        self._start_timer()

    def _self_destruct(self) -> None:
        self.bot.del_conversation(self.key)

    def handle_request(self, request: dict) -> dict:
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
        if self.stateful:
            utterance = [[utterance], [self.key]]
        else:
            utterance = [[utterance]]

        agent_response: list = self.agent(*utterance)

        return agent_response

    def _generate_response(self, response: dict, request: dict) -> dict:
        response_template = deepcopy(self.response_template)
        response_template['sessionAttributes']['sessionId'] = request['session']['sessionId']

        for key, value in response_template.items():
            if key not in response.keys():
                response[key] = value

        return response

    def _handle_intent(self, request: dict) -> dict:
        intent_name = self.bot.config['intent_name']
        slot_name = self.bot.config['slot_name']

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
        response = {
            'response': {
                'shouldEndSession': False,
                'outputSpeech': {
                    'type': 'PlainText',
                    'text': self.bot.config['start_message']
                },
                'card': {
                    'type': 'Simple',
                    'content': self.bot.config['start_message']
                }
            }
        }

        response = self._generate_response(response, request)

        return response

    def _handle_end(self, request: dict) -> dict:
        response = {}
        self._self_destruct()
        return response

    def _handle_unsupported(self, request: dict) -> dict:
        response = {
            'response': {
                'shouldEndSession': False,
                'outputSpeech': {
                    'type': 'PlainText',
                    'text': self.bot.config['unsupported_message']
                },
                'card': {
                    'type': 'Simple',
                    'content': self.bot.config['unsupported_message']
                }
            }
        }

        response = self._generate_response(response, request)

        return response
