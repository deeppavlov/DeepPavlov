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
from urllib.parse import urljoin

from requests import Session

from deeppavlov.utils.wrapper import BaseConversation

log = getLogger(__name__)


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

        self._start_timer()

    def handle_request(self, request: dict):
        activity_type = request['type']
        activity_id = request['id']
        log.debug(f'Received activity. Type: {activity_type}, id: {activity_id}')

        if activity_type in self._handled_activities.keys():
            self._handled_activities[activity_type](request)
        else:
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
        self._send_plain_text(self._config['start_message'])
