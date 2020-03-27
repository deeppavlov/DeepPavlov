from typing import NamedTuple


class GobotAttnHyperParams(NamedTuple):
    # todo migrate to dataclasses?
    key_size: int
    token_size: int
    window_size: int
    use_action_as_key: bool
    use_intent_as_key: bool

class GobotAttnParams(NamedTuple):
    max_num_tokens: int
    hidden_size: int
    token_size: int
    key_size: int
    type: str
    projected_align: bool
    depth: int
    action_as_key: bool
    intent_as_key: bool