import threading
from copy import deepcopy
from urllib.parse import urljoin

import requests

from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


class Conversation:
    def __init__(self, bot, model, activity: dict, conversation_key):
        self.bot = bot
        self.model = model
        self.key = conversation_key

        self.bot_id = activity['recipient']['id']
        self.bot_name = activity['recipient']['name']
        self.service_url = activity['serviceUrl']
        self.channel_id = activity['channelId']
        self.conversation_id = activity['conversation']['id']

        self.out_gateway = OutGateway(self)

        self.rich_content = self.bot.config['rich_content']
        self.stateful = self.bot.config['stateful']
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

        if activity_type in self.handled_activities.keys():
            self.handled_activities[activity_type](activity)
        else:
            log.warning(f'Unsupported activity type: {activity_type}, activity id: {activity_id}')

        self._rearm_self_destruct()

    def _infer(self, raw_observation: [tuple, str]):
        if self.stateful:
            content = tuple([raw_observation]) if not isinstance(raw_observation, tuple) else raw_observation
            observation = [tuple([self.key]) + content]
        else:
            observation = [raw_observation]

        prediction = self.model(observation)

        return prediction

    def _send_infer_results(self, prediction: list, in_activity: dict):
        pred = prediction[0]

        if self.rich_content:
            for rich_message in pred:
                if rich_message['type'] == 'text':
                    self.out_gateway.send_plain_text(rich_message['value'], in_activity)
                elif rich_message['type'] == 'button':
                    self.out_gateway.send_buttons(rich_message['value'], in_activity)
        else:
            self.out_gateway.send_plain_text(str(pred), in_activity)

    def _handle_usupported(self, in_activity: dict):
        activity_type = in_activity['type']
        self.out_gateway.send_plain_text(f'Unsupported kind of {activity_type} activity!')
        log.warn(f'Recived message with unsupported type: {str(in_activity)}')

    def _handle_message(self, in_activity: dict):
        if 'text' in in_activity.keys():
            in_text = in_activity['text']

            if len(self.in_x) > 1:
                if not self.multiargument_initiated:
                    self.multiargument_initiated = True
                    self.expect[:] = list(self.in_x)
                    self.out_gateway.send_plain_text(f'Please, send {self.expect.pop(0)}')
                else:
                    self.buffer.append(in_text)

                    if self.expect:
                        self.out_gateway.send_plain_text(f'Please, send {self.expect.pop(0)}', in_activity)
                    else:
                        prediction = self._infer(tuple(self.buffer))
                        self._send_infer_results(prediction, in_activity)

                        self.buffer = []
                        self.expect[:] = list(self.in_x)
                        self.out_gateway.send_plain_text(f'Please, send {self.expect.pop(0)}', in_activity)
            else:
                prediction = self._infer(in_text)
                self._send_infer_results(prediction, in_activity)
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

    def _send_activity(self, out_activity: dict, in_activity: dict = None):
        service_url = self.service_url

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

        log.debug(f'Sent activity to the MSBotFramework server. '
                  f'Response code: {response.status_code}, response contents: {response.json()}')

    def send_plain_text(self, text: str, in_activity: dict = None):
        out_activity = deepcopy(self.activity_template)
        out_activity['type'] = 'message'
        out_activity['text'] = text
        self._send_activity(out_activity, in_activity)

    def send_buttons(self, buttons: list, in_activity: dict = None):
        out_activity = deepcopy(self.activity_template)
        out_activity['type'] = 'message'

        # Creating RichCard with CardActions(buttons) with postBack value return
        rich_card = {
            'buttons': []
        }

        for button in buttons:
            card_action = {}
            card_action['type'] = 'postBack'
            card_action['title'] = button['name']
            card_action['value'] = button['callback']
            rich_card['buttons'].append(card_action)

        attachments = [
            {
                "contentType": "application/vnd.microsoft.card.thumbnail",
                "content": rich_card
            }
        ]

        out_activity['attachments'] = attachments
        self._send_activity(out_activity, in_activity)
