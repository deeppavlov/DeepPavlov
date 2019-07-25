from typing import Optional, Union

import json

from deeppavlov.skills.dsl_skill.utils import UserId


class UserContext:
    """
    UserContext object stores information that the current skill currently knows about the user.

    Args:
        user_id: int, id of user
        message: str, current message
        current_state: str or None, current user state
        payload: dict or str, custom payload dictionary, or a JSON-serialized string of such dictionary
    """

    def __init__(
            self,
            user_id: Optional[UserId] = None,
            message: Optional[str] = None,
            current_state: Optional[str] = None,
            payload: Optional[Union[dict, str]] = None,
    ):
        self.user_id = user_id
        self.message = message
        self.current_state = current_state

        # some custom data added by skill creator
        self.payload = payload
        if payload == '' or payload is None:
            self.payload = {}
        elif isinstance(payload, str):
            self.payload = json.loads(payload)
