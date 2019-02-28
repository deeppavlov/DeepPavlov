import threading
from logging import getLogger
from urllib.parse import urljoin

import requests

from deeppavlov.core.agent.rich_content import RichMessage

log = getLogger(__name__)


class Conversation:
    def __init__(self, bot, agent, activity: dict, conversation_key):
        self.bot = bot
        self.agent = agent
        self.key = conversation_key

        self.bot_id = activity['recipient']['id']
        self.bot_name = activity['recipient']['name']
        self.service_url = activity['serviceUrl']
        self.channel_id = activity['channelId']
        self.conversation_id = activity['conversation']['id']

        self.out_gateway = OutGateway(self)
        self.stateful = self.bot.config['stateful']

        self.conversation_lifetime = self.bot.config['conversation_lifetime']
        self.timer = None
        self._start_timer()

        if self.channel_id not in self.bot.http_sessions.keys() or not self.bot.http_sessions['self.channel_id']:
            self.bot.http_sessions['self.channel_id'] = requests.Session()

        self.http_session = self.bot.http_sessions['self.channel_id']

        self.handled_activities = {
            'message': self._handle_message
        }

    def _start_timer(self):
        self.timer = threading.Timer(self.conversation_lifetime, self._self_destruct)
        self.timer.start()

    def _rearm_self_destruct(self):
        self.timer.cancel()
        self._start_timer()

    def _self_destruct(self):
        self.bot.del_conversation(self.key)

    def handle_activity(self, activity: dict):
        activity_type = activity['type']
        activity_id = activity['id']
        log.debug(f'Received activity. Type: {activity_type}, id: {activity_id}')

        if activity_type in self.handled_activities.keys():
            self.handled_activities[activity_type](activity)
        else:
            log.warning(f'Unsupported activity type: {activity_type}, activity id: {activity_id}')

        self._rearm_self_destruct()

    def _act(self, utterance: str):
        if self.stateful:
            utterance = [[utterance], [self.key]]
        else:
            utterance = [[utterance]]

        prediction = self.agent(*utterance)

        return prediction

    def _send_infer_results(self, response: RichMessage, in_activity: dict):
        ms_bf_response = response.ms_bot_framework()
        for out_activity in ms_bf_response:
            if out_activity:
                self.out_gateway.send_activity(out_activity, in_activity)

    def _handle_usupported(self, in_activity: dict):
        activity_type = in_activity['type']
        self.out_gateway.send_plain_text(f'Unsupported kind of {activity_type} activity!')
        log.warn(f'Received message with unsupported type: {str(in_activity)}')

    def _handle_message(self, in_activity: dict):
        if 'text' in in_activity.keys():
            in_text = in_activity['text']
            agent_response = self._act(in_text)
            if agent_response:
                response = agent_response[0]
                self._send_infer_results(response, in_activity)
        else:
            self._handle_usupported(in_activity)


class OutGateway:
    def __init__(self, conversation: Conversation):
        self.conversation = conversation
        self.service_url = self.conversation.service_url
        self.activity_template = {
            'from': {
                'id': self.conversation.bot_id,
                'name': self.conversation.bot_name
            },
            'conversation': {
                'id': self.conversation.conversation_id
            }
        }

    def send_activity(self, out_activity: dict, in_activity: dict = None):
        service_url = self.service_url

        for key, value in self.activity_template.items():
            out_activity[key] = value

        if in_activity:
            try:
                service_url = in_activity['serviceUrl']
            except KeyError:
                pass

            try:
                out_activity['recepient']['id'] = in_activity['from']['id']
            except KeyError:
                pass

            try:
                out_activity['conversation']['name'] = in_activity['conversation']['name']
            except KeyError:
                pass

            try:
                out_activity['recepient']['name'] = in_activity['from']['name']
            except KeyError:
                pass

        url = urljoin(service_url, f"v3/conversations/{self.conversation.conversation_id}/activities")

        authorization = f"{self.conversation.bot.access_info['token_type']} " \
                        f"{self.conversation.bot.access_info['access_token']}"
        headers = {
            'Authorization': authorization,
            'Content-Type': 'application/json'
        }

        response = self.conversation.http_session.post(
            url=url,
            json=out_activity,
            headers=headers)

        try:
            response_json_str = str(response.json())
        except Exception:
            response_json_str = ''

        log.debug(f'Sent activity to the MSBotFramework server. '
                  f'Response code: {response.status_code}, response contents: {response_json_str}')
