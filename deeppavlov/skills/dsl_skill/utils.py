from typing import Callable, Union, Tuple, Optional

UserId = Union[str, int]
ResponseType = Tuple[str, float, Optional[str]]


def expand_arguments(func: Callable):
    if func.__code__.co_argcount == 0:
        return lambda msg, context: func()
    elif func.__code__.co_argcount == 1:
        return lambda msg, context: func(msg)
    else:
        return func

