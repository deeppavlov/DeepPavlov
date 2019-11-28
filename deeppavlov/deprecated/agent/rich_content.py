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
from typing import Union


class RichItem(metaclass=ABCMeta):
    """Base class for rich content elements.

    Every rich content element
    is presumed to return its state (including state of nested controls)
    at least in json format (mandatory) as well as in the formats compatible
    with other channels.
    """

    @abstractmethod
    def json(self) -> Union[list, dict]:
        """Returns json compatible state of the control instance including
        its nested controls.

        Returns:
            control: Json representation of control state.
        """
        pass

    def ms_bot_framework(self):
        """Returns MS Bot Framework compatible state of the control instance
        including its nested controls.

        Returns:
            control: MS Bot Framework representation of control state.
        """
        return None

    def telegram(self):
        """Returns Telegram compatible state of the control instance
        including its nested controls.

        Returns:
            control: Telegram representation of control state.
        """
        return None

    def alexa(self):
        """Returns Amazon Alexa compatible state of the control instance
        including its nested controls.

        Returns:
            control: Amazon Alexa representation of control state.
        """
        return None


class RichControl(RichItem, metaclass=ABCMeta):
    """Base class for rich controls.

    Rich control can be a button, buttons box, plain text, image, etc.
    All rich control classes should be derived from RichControl.

    Args:
        control_type: Name of the rich control type.

    Attributes:
        control_type: Name of the rich control type.
        content: Arbitrary used control content holder.
        control_json: Control json representation template, which
            contains control type and content fields.
    """

    def __init__(self, control_type: str) -> None:
        self.control_type: str = control_type
        self.content = None
        self.control_json: dict = {'type': control_type, 'content': None}

    def __str__(self) -> str:
        return ''


class RichMessage(RichItem):
    """Container for rich controls.

    All rich content elements returned by agent as a result of single
    inference should be embedded into RichMessage instance in the order
    these elements should be displayed.

    Attributes:
        controls: Container for RichControl instances.
    """

    def __init__(self) -> None:
        self.controls: list = []

    def __str__(self) -> str:
        result = '\n'.join(filter(bool, map(str, self.controls)))
        return result

    def add_control(self, control: RichControl):
        """Adds RichControl instance to RichMessage.

        Args:
            control: RichControl instance.
        """
        self.controls.append(control)

    def json(self) -> list:
        """Returns list of json compatible states of the RichMessage instance
        nested controls.

        Returns:
            json_controls: Json representation of RichMessage instance
                nested controls.
        """
        json_controls = [control.json() for control in self.controls]
        return json_controls

    def ms_bot_framework(self) -> list:
        """Returns list of MS Bot Framework compatible states of the
        RichMessage instance nested controls.

        Returns:
            ms_bf_controls: MS Bot Framework representation of RichMessage instance
                nested controls.
        """
        ms_bf_controls = [control.ms_bot_framework() for control in self.controls]
        return ms_bf_controls

    def telegram(self) -> list:
        """Returns list of Telegram compatible states of the RichMessage
        instance nested controls.

        Returns:
            telegram_controls: Telegram representation of RichMessage instance nested
                controls.
        """
        telegram_controls = [control.telegram() for control in self.controls]
        return telegram_controls

    def alexa(self) -> list:
        """Returns list of Amazon Alexa compatible states of the RichMessage
        instance nested controls.

        Returns:
            alexa_controls: Amazon Alexa representation of RichMessage instance nested
                controls.
        """
        alexa_controls = [control.alexa() for control in self.controls]
        return alexa_controls
