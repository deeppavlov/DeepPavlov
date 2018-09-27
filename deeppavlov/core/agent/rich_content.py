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
    """Parent class for rich content elements.

    Every rich content element
    is presumed to return its state (including state of nested controls)
    at least in json format (mandatory) as well as in the formats compatible
    with other channels.
    """
    @abstractmethod
    def json(self) -> [list, dict]:
        """Returns json compatible state of the control instance including
        its nested controls.

        Returns:
            control (list, dict): json representation of control state
        """
        return None

    def ms_bot_framework(self):
        """Returns MS Bot Framework compatible state of the control instance
        including its nested controls.

        Returns:
            control: MS Bot Framework representation of control state
        """
        return None

    def telegram(self):
        """Returns Telegram compatible state of the control instance
        including its nested controls.

        Returns:
            control: Telegram representation of control state
        """
        return None


class RichControl(RichItem, metaclass=ABCMeta):
    """Parent class for rich controls.

    Rich control can be a button, buttons
    box, plain text, image, etc. Each control class should be derived from
    RichControl.

    Args:
        control_type (str): Name of the rich control type.

    Attributes:
        control_type (str): Name of the rich control type.
        content: Arbitrary used control content holder.
        control_json (dict): control json representation template, which
            contains control type and content fields.
    """
    def __init__(self, control_type: str):
        self.control_type = control_type
        self.content = None
        self.control_json = {'type': control_type, 'content': None}


class RichMessage(RichItem):
    """Container for rich controls.

    All rich content elements returned by agent as a result of single
    inference should be embedded into RichMessage instance in the order
    these elements should be displayed.

    Attributes:
        controls (list): Container for RichControl instances.
    """

    def __init__(self):
        self.controls = []

    def add_control(self, control: RichControl):
        """Adds RichControl to RichMessage.

        Args:
            control (RichControl): Rich control instance.
        """
        self.controls.append(control)

    def json(self):
        """Returns json compatible state of the RichMessage instance including
        its nested controls.

        Returns:
            json_controls (list): json representation of RichMessage and
            embedded rich controls.
        """
        json_controls = [control.json() for control in self.controls]
        return json_controls

    def ms_bot_framework(self):
        """Returns MS Bot Framework compatible state of the RichMessage instance
        including its nested controls.

        Returns:
            control: MS Bot Framework representation of RichMessage state
        """
        ms_bf_controls = [control.ms_bot_framework() for control in self.controls]
        return ms_bf_controls

    def telegram(self):
        """Returns Telegram compatible state of the RichMessage instance
        including its nested controls.

        Returns:
            control: Telegram representation of RichMessage state
        """
        telegram_controls = [control.telegram() for control in self.controls]
        return telegram_controls
