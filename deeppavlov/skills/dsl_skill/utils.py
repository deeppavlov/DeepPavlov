from typing import Union, NamedTuple

UserId = Union[str, int]


class SkillResponse(NamedTuple):
    response: str
    confidence: float
