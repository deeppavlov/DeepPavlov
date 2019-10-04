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
from urllib.parse import urljoin

import requests

from deeppavlov.deprecated.agent import RichMessage
from deeppavlov.utils.bot import BaseConversation

log = getLogger(__name__)


class MSConversation(BaseConversation):
    http_sessions = dict()

    def __init__(self, config, agent, activity: dict, conversation_key, self_destruct_callback: callable,
                 get_headers_callback: callable):
        super(MSConversation, self).__init__(config, agent, conversation_key, self_destruct_callback)
        self._get_headers_callback = get_headers_callback

        self.bot_id = activity['recipient']['id']
        self.bot_name = activity['recipient']['name']
        self.service_url = activity['serviceUrl']
        self.channel_id = activity['channelId']
        self.conversation_id = activity['conversation']['id']

        if self.channel_id not in MSConversation.http_sessions:
            MSConversation.http_sessions[self.channel_id] = requests.Session()

        self.http_session = MSConversation.http_sessions[self.channel_id]

        self.handled_activities = {
            'message': self._handle_message
        }

        self.activity_template = {
            'from': {
                'id': self.bot_id,
                'name': self.bot_name
            },
            'conversation': {
                'id': self.conversation_id
            }
        }

        self._start_timer()

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
                self.send_activity(out_activity, in_activity)

    def _handle_usupported(self, in_activity: dict):
        activity_type = in_activity['type']
        self.send_plain_text(f'Unsupported kind of {activity_type} activity!')
        log.warning(f'Received message with unsupported type: {str(in_activity)}')

    def _handle_message(self, in_activity: dict):
        if 'text' in in_activity.keys():
            in_text = in_activity['text']
            agent_response = self._act(in_text)
            if agent_response:
                response = agent_response[0]
                self._send_infer_results(response, in_activity)
        else:
            self._handle_usupported(in_activity)

    def send_activity(self, out_activity: dict, in_activity: dict = None):
        service_url = self.service_url
        for key, value in self.activity_template.items():
            out_activity[key] = value

        if in_activity:
            service_url = in_activity.get('serviceUrl', service_url)

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

        response = self.http_session.post(
            url=url,
            json=out_activity,
            headers=self._get_headers_callback())

        try:
            response_json_str = str(response.json())
        except Exception:
            response_json_str = ''

        log.debug(f'Sent activity to the MSBotFramework server. '
                  f'Response code: {response.status_code}, response contents: {response_json_str}')

    def send_plain_text(self, text: str):
        raise NotImplementedError
