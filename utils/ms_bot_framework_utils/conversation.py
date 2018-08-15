import requests
from urllib.parse import urljoin

from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


class Conversation:
    def __init__(self, bot, activity: dict):
        self.bot = bot
        self.bot_id = activity['recipient']['id']
        self.bot_name = activity['recipient']['name']
        self.service_url = activity['serviceUrl']
        self.channel_id = activity['channelId']
        self.conversation_id = activity['conversation']['id']
        self.buffer = []
        self.expect = []
        self.multiargument_initiated = False

        if self.channel_id not in self.bot.http_sessions.keys() or not self.bot.http_sessions['self.channel_id']:
            self.bot.http_sessions['self.channel_id'] = requests.Session()

        self.http_session = self.bot.http_sessions['self.channel_id']

        self.handled_activities = {
            'message': self._handle_message
        }

    def handle_activity(self, activity: dict):
        activity_type = activity['type']
        activity_id = activity['id']
        log.debug(f'Received activity. Type: {activity_type}, id: {activity_id}')

        if activity_type in self.handled_activities.keys():
            self.handled_activities[activity_type](activity)

    def _send_activity(self, url: str, out_activity: dict):
        authorization = f"{self.bot.access_info['token_type']} {self.bot.access_info['access_token']}"
        headers = {
            'Authorization': authorization,
            'Content-Type': 'application/json'
        }

        response = self.http_session.post(
            url=url,
            json=out_activity,
            headers=headers)

        log.debug(f'Sent activity to the MSBotFramework server. '
                  f'Response code: {response.status_code}, response contents: {response.json()}')

    def _send_message(self, message_text: str, in_activity: dict = None):
        service_url = self.service_url

        out_activity = {
            'type': 'message',
            'from': {
                'id': self.bot_id,
                'name': self.bot_name
            },
            'conversation': {
                'id': self.conversation_id
            },
            'text': message_text
        }

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

        url = urljoin(service_url, f"v3/conversations/{self.conversation_id}/activities")

        self._send_activity(url, out_activity)

    def _handle_message(self, in_activity: dict):

        def handle_text():
            in_text = in_activity['text']

            if len(self.bot.model.in_x) > 1:
                if not self.multiargument_initiated:
                    self.multiargument_initiated = True
                    self.expect[:] = list(self.bot.model.in_x)
                    self._send_message(f'Please, send {self.expect.pop(0)}')
                else:
                    self.buffer.append(in_text)

                    if self.expect:
                        self._send_message(f'Please, send {self.expect.pop(0)}', in_activity)
                    else:
                        pred = self.bot.model([tuple(self.buffer)])
                        out_text = str(pred[0])
                        self._send_message(out_text, in_activity)

                        self.buffer = []
                        self.expect[:] = list(self.bot.model.in_x)
                        self._send_message(f'Please, send {self.expect.pop(0)}', in_activity)
            else:
                pred = self.bot.model([in_text])
                out_text = str(pred[0])
                self._send_message(out_text, in_activity)

        def handle_unsupported():
            self._send_message('Unsupported message type!', in_activity)
            log.warn(f'Recived message with unsupported type: {str(in_activity)}')

        if 'text' in in_activity.keys():
            handle_text()
        else:
            handle_unsupported()
