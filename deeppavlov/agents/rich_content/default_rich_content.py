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
from deeppavlov.core.agent.rich_content import RichControl


class PlainText(RichControl):
    def __init__(self, text):
        super(PlainText, self).__init__('plain_text')
        self.content = text

    def json(self):
        self.control_json['content'] = self.content
        return self.control_json

    def ms_bot_framework(self):
        # Creating MS Bot Framework activity blank with "text" populated
        out_activity = {}
        out_activity['type'] = 'message'
        out_activity['text'] = self.content
        return out_activity


class Button(RichControl):
    def __init__(self, name: str, callback: str):
        super(Button, self).__init__('button')
        self.name = name
        self.callback = callback

    def json(self):
        content = {}
        content['name'] = self.name
        content['callback'] = self.callback
        self.control_json['content'] = content
        return self.control_json

    def ms_bot_framework(self):
        # Creating MS Bot Framework CardAction (button) with postBack value return
        card_action = {}
        card_action['type'] = 'postBack'
        card_action['title'] = self.name
        card_action['value'] = self.callback = self.callback
        return card_action


class ButtonsFrame(RichControl):
    def __init__(self, text: [str, None] = None):
        super(ButtonsFrame, self).__init__('buttons_frame')
        self.text = text
        self.content = []

    def add_button(self, button: Button):
        self.content.append(button)

    def json(self):
        content = {}

        if self.text:
            content['text'] = self.text

        content['controls'] = [control.json() for control in self.content]

        self.control_json['content'] = content

        return self.control_json

    def ms_bot_framework(self):
        # Creating MS Bot Framework activity blank with RichCard in "attachments" populated with CardActions
        rich_card = {}

        buttons = [button.ms_bot_framework() for button in self.content]
        rich_card['buttons'] = buttons

        if self.text:
            rich_card['title'] = self.text

        attachments = [
            {
                "contentType": "application/vnd.microsoft.card.thumbnail",
                "content": rich_card
            }
        ]

        out_activity = {}
        out_activity['type'] = 'message'
        out_activity['attachments'] = attachments

        return out_activity
