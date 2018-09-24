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
from abc import ABCMeta, abstractmethod


class RichItem(metaclass=ABCMeta):
    @abstractmethod
    def json(self) -> [list, dict]:
        return None

    def ms_bot_framework(self):
        return None

    def telegram(self):
        return None


class RichControl(RichItem, metaclass=ABCMeta):
    def __init__(self, control_type: str):
        self.control_type = control_type
        self.content = None
        self.control_json = {'type': control_type}


class RichMessage(RichItem):
    def __init__(self):
        self.controls = []

    def add_control(self, control: RichControl):
        self.controls.append(control)

    def json(self):
        json_controls = [control.json() for control in self.controls]
        return json_controls

    def ms_bot_framework(self):
        ms_bf_controls = [control.ms_bot_framework() for control in self.controls]
        return ms_bf_controls

    def telegram(self):
        telegram_controls = [control.telegram() for control in self.controls]
        return telegram_controls


class PlainText(RichControl):
    def __init__(self, text):
        super(PlainText, self).__init__('plain_text')
        self.content = text

    def json(self):
        self.control_json['content'] = self.content
        return self.control_json

    def ms_bot_framework(self):
        out_activity = {}
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


class ButtonsFrame(RichControl):
    def __init__(self):
        super(ButtonsFrame, self).__init__('buttons_frame')
        self.content = []

    def add_button(self, button: Button):
        self.content.append(button)

    def json(self):
        json_content = [control.json() for control in self.content]
        return json_content
