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

from typing import Optional

from deeppavlov.deprecated.agent import RichControl


class PlainText(RichControl):
    """Plain text message as a rich control.

    Args:
        text: Text of the message.

    Attributes:
        content: Text of the message.
    """

    def __init__(self, text: str) -> None:
        super(PlainText, self).__init__('plain_text')
        self.content: str = text

    def __str__(self) -> str:
        return self.content

    def json(self) -> dict:
        """Returns json compatible state of the PlainText instance.

        Returns:
            control_json: Json representation of PlainText state.
        """
        self.control_json['content'] = self.content
        return self.control_json

    def ms_bot_framework(self) -> dict:
        """Returns MS Bot Framework compatible state of the PlainText instance.

        Creating MS Bot Framework activity blank with "text" field populated.

        Returns:
            out_activity: MS Bot Framework representation of PlainText state.
        """
        out_activity = {}
        out_activity['type'] = 'message'
        out_activity['text'] = self.content
        return out_activity

    def alexa(self) -> dict:
        """Returns Amazon Alexa compatible state of the PlainText instance.

        Creating Amazon Alexa response blank with populated "outputSpeech" and
        "card sections.

        Returns:
            response: Amazon Alexa representation of PlainText state.
        """
        response = {
            'response': {
                'shouldEndSession': False,
                'outputSpeech': {
                    'type': 'PlainText',
                    'text': self.content},
                'card': {
                    'type': 'Simple',
                    'content': self.content
                }
            }
        }

        return response


class Button(RichControl):
    """Button with plain text callback.

    Args:
        name: Displayed name of the button.
        callback: Plain text returned as callback when button pressed.

    Attributes:
        name: Displayed name of the button.
        callback: Plain text returned as callback when button pressed.
    """

    def __init__(self, name: str, callback: str) -> None:
        super(Button, self).__init__('button')
        self.name: str = name
        self.callback: str = callback

    def json(self) -> dict:
        """Returns json compatible state of the Button instance.

        Returns:
            control_json: Json representation of Button state.
        """
        content = {}
        content['name'] = self.name
        content['callback'] = self.callback
        self.control_json['content'] = content
        return self.control_json

    def ms_bot_framework(self) -> dict:
        """Returns MS Bot Framework compatible state of the Button instance.

        Creates MS Bot Framework CardAction (button) with postBack value return.

        Returns:
            control_json: MS Bot Framework representation of Button state.
        """
        card_action = {}
        card_action['type'] = 'postBack'
        card_action['title'] = self.name
        card_action['value'] = self.callback = self.callback
        return card_action


class ButtonsFrame(RichControl):
    """ButtonsFrame is a container for several Buttons objects.

    ButtonsFrame embeds several Buttons and allows to post them
    in one channel message.

    Args:
        text: Text displayed with embedded buttons.

    Attributes:
        text: Text displayed with embedded buttons.
        content: Container with Button objects.
    """

    def __init__(self, text: Optional[str] = None) -> None:
        super(ButtonsFrame, self).__init__('buttons_frame')
        self.text: [str, None] = text
        self.content: list = []

    def add_button(self, button: Button):
        """Adds Button instance to RichMessage.

        Args:
            button: Button instance.
        """
        self.content.append(button)

    def json(self) -> dict:
        """Returns json compatible state of the ButtonsFrame instance.

        Returns json compatible state of the ButtonsFrame instance including
        all nested buttons.

        Returns:
            control_json: Json representation of ButtonsFrame state.
        """
        content = {}

        if self.text:
            content['text'] = self.text

        content['controls'] = [control.json() for control in self.content]

        self.control_json['content'] = content

        return self.control_json

    def ms_bot_framework(self) -> dict:
        """Returns MS Bot Framework compatible state of the ButtonsFrame instance.

        Creating MS Bot Framework activity blank with RichCard in "attachments". RichCard
        is populated with CardActions corresponding buttons embedded in ButtonsFrame.

        Returns:
            control_json: MS Bot Framework representation of ButtonsFrame state.
        """
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
