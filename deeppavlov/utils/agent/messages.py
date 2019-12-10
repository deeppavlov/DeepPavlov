from typing import Any, Dict


class MessageBase:
    agent_name: str
    msg_type: str

    def __init__(self, msg_type: str, agent_name: str):
        self.msg_type = msg_type
        self.agent_name = agent_name

    @classmethod
    def from_json(cls, message_json):
        return cls(**message_json)

    def to_json(self) -> dict:
        return self.__dict__


class ServiceTaskMessage(MessageBase):
    payload: Dict

    def __init__(self, agent_name: str, payload: Dict) -> None:
        super().__init__('service_task', agent_name)
        self.payload = payload


class ServiceResponseMessage(MessageBase):
    response: Any
    task_id: str

    def __init__(self, task_id: str, agent_name: str, response: Any) -> None:
        super().__init__('service_response', agent_name)
        self.task_id = task_id
        self.response = response


def get_service_task_message(message_json: dict) -> ServiceTaskMessage:
    """Creates instance of ServiceTaskMessage class using its json representation.

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
