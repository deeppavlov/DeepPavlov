import threading
from urllib.parse import urljoin

import requests

from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


class Conversation:
    def __init__(self, bot, model, activity: dict, conversation_key
                 # , conversation_lifetime: int
                 ):
        self.bot = bot
        self.model = model
        self.key = conversation_key
        self.out_gateway = OutGateway(self)

        self.bot_id = activity['recipient']['id']
        self.bot_name = activity['recipient']['name']
        self.service_url = activity['serviceUrl']
        self.channel_id = activity['channelId']
        self.conversation_id = activity['conversation']['id']

        self.rich_content = self.bot.config['rich_content']
        self.stateful = self.bot.config['stateful']
        self.state = None
        self.in_x = self.model.in_x[1:] if self.stateful else self.model.in_x

        self.buffer = []
        self.expect = []
        self.multiargument_initiated = False

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

        # REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        log.debug(f'\nACTIVITY:\n{activity}\n')

        if activity_type in self.handled_activities.keys():
            self.handled_activities[activity_type](activity)
        else:
            log.warning(f'Unsupported activity type: {activity_type}, activity id: {activity_id}')

        self._rearm_self_destruct()

    def _handle_usupported(self, activity: dict):
        activity_type = activity['type']
        self.out_gateway.send_message(f'Unsupported kind of {activity_type} activity!')
        log.warn(f'Recived message with unsupported type: {str(in_activity)}')

    def _send_infer_results(self, pre):

    def _handle_message(self, in_activity: dict):
        if 'text' in in_activity.keys():
            in_text = in_activity['text']
        else:
            self._handle_usupported(in_activity)



        """
        def infer(content: [tuple, str]):
            if self.stateful:
                content = tuple([content]) if isinstance(content, str) else content
                observation = [tuple([self.state]) + content]
            else:
                observation = [content]
            return self.model.infer(observation)

        def handle_text():
            in_text = in_activity['text']

            if len(self.in_x) > 1:
                if not self.multiargument_initiated:
                    self.multiargument_initiated = True
                    self.expect[:] = list(self.in_x)
                    self.out_gateway.send_message(f'Please, send {self.expect.pop(0)}')
                else:
                    self.buffer.append(in_text)

                    if self.expect:
                        self.out_gateway.send_message(f'Please, send {self.expect.pop(0)}', in_activity)
                    else:
                        pred = infer(tuple(self.buffer))
                        out_text = str(pred[0])
                        self.out_gateway.send_message(out_text, in_activity)

                        self.buffer = []
                        self.expect[:] = list(self.in_x)
                        self.out_gateway.send_message(f'Please, send {self.expect.pop(0)}', in_activity)
            else:
                pred = infer(in_text)
                out_text = str(pred[0])
                self.out_gateway.send_message(out_text, in_activity)

        def handle_unsupported():
            self.out_gateway.send_message('Unsupported message type!', in_activity)
            log.warn(f'Recived message with unsupported type: {str(in_activity)}')

        if 'text' in in_activity.keys():
            handle_text()
        else:
            handle_unsupported()
        """

    def _infer(self, content: [tuple, str]):
        if self.stateful:
            content = tuple([content]) if not isinstance(content, tuple) else content
            observation = [tuple([self.state]) + content]
        else:
            observation = [content]

        prediction = self.model(observation)

        if self.stateful:
            self.state = prediction['state']

        return prediction


class OutGateway:
    def __init__(self, conversation: Conversation):
        self.conversation = conversation

    def _send_activity(self, url: str, out_activity: dict):
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

        log.debug(f'Sent activity to the MSBotFramework server. '
                  f'Response code: {response.status_code}, response contents: {response.json()}')

    def send_message(self, message_text: str, in_activity: dict = None):
        service_url = self.conversation.service_url

        card_action_1 = {
            "title": "Test_1",
            "type": "imBack",
            "value": "Test_a_button_1"
        }

        card_action_2 = {
            "title": "Test_2",
            "type": "imBack",
            "value": "Test_a_button_2"
        }

        rich_card = {
            "buttons": [card_action_1, card_action_2]
        }

        attachments = [
            {
                "contentType": "application/vnd.microsoft.card.thumbnail",
                "content": rich_card
            }
        ]

        out_activity = {
            'type': 'message',
            'from': {
                'id': self.conversation.bot_id,
                'name': self.conversation.bot_name
            },
            'conversation': {
                'id': self.conversation.conversation_id
            },
            'text': message_text,
            'attachments': attachments
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

        url = urljoin(service_url, f"v3/conversations/{self.conversation.conversation_id}/activities")

        self._send_activity(url, out_activity)
