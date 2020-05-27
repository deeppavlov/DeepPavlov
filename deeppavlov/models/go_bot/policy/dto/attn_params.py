from typing import NamedTuple


class GobotAttnParams(NamedTuple):
    """
    the DTO-like class that stores the attention mechanism configuration params.
    """
    max_num_tokens: int
    hidden_size: int
    token_size: int
    key_size: int
    type_: str
    projected_align: bool
    depth: int
    action_as_key: bool
    intent_as_key: bool
