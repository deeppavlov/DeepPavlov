# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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
"""Module contains classes defining messages received and sent by service via RabbitMQ message broker.

The classes created to document the DeepPavlov Agent API and should match the corresponding classes
from https://github.com/deepmipt/dp-agent/blob/master/core/transport/messages.py

"""

from typing import Any


class MessageBase:
    agent_name: str
    msg_type: str

    def __init__(self, msg_type: str, agent_name: str) -> None:
        self.msg_type = msg_type
        self.agent_name = agent_name

    @classmethod
    def from_json(cls, message_json: dict):
        return cls(**message_json)

    def to_json(self) -> dict:
        return self.__dict__


class ServiceTaskMessage(MessageBase):
    payload: dict

    def __init__(self, agent_name: str, payload: dict) -> None:
        super().__init__('service_task', agent_name)
        self.payload = payload


class ServiceResponseMessage(MessageBase):
    response: Any
    task_id: str

    def __init__(self, task_id: str, agent_name: str, response: Any) -> None:
        super().__init__('service_response', agent_name)
        self.task_id = task_id
        self.response = response


class ServiceErrorMessage(MessageBase):
    formatted_exc: str

    def __init__(self, task_id: str, agent_name: str, formatted_exc: str) -> None:
        super().__init__('error', agent_name)
        self.task_id = task_id
        self.formatted_exc = formatted_exc

    @property
    def exception(self) -> Exception:
        return Exception(self.formatted_exc)


def get_service_task_message(message_json: dict) -> ServiceTaskMessage:
    """Creates an instance of ServiceTaskMessage class using its json representation.

    Args:
        message_json: Dictionary with class fields.

    Returns:
        New ServiceTaskMessage instance.

    Raises:
        ValueError if dict with instance fields isn't from an instance of ServiceTaskMessage class.

    """
    message_type = message_json.pop('msg_type')

    if message_type != 'service_task':
        raise TypeError(f'Unknown transport message type: {message_type}')

    return ServiceTaskMessage.from_json(message_json)
