# encoding: utf-8

import re
from abc import ABCMeta
from collections import defaultdict
from functools import partial
from itertools import zip_longest
from typing import List, Optional, Union, Dict, Callable, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.skills.dsl_skill.handlers import Handler, RegexHandler
from deeppavlov.skills.dsl_skill.utils import expand_arguments, ResponseType


class DSLMeta(ABCMeta):
    """
    This metaclass is used for create a skill.

    Example:

    .. code:: python

            class ExampleSkill(metaclass=ZDialog):
                @ZDialog.handler(commands=["hello", "hey"], state="greeting")
                def __greeting(message: str):
                    ...
    """
    skill_collection: Dict[str, 'DSLMeta'] = {}

    def __init__(cls, name: str, bases, namespace, **kwargs):
        super(DSLMeta, cls).__init__(name, bases, namespace, **kwargs)
        cls.name = name

        # Attribute cls.state_to_handler is dict with states as keys and lists of Handler objects as values
        cls.state_to_handler = defaultdict(list)
        # Handlers that can be activated from any state
        cls.universal_handlers = []

        handlers = [attribute for attribute in namespace.values() if isinstance(attribute, Handler)]

        for handler in handlers:
            if handler.state is None:
                cls.universal_handlers.append(handler)
            else:
                cls.state_to_handler[handler.state].append(handler)

        cls.handle = partial(DSLMeta.__handle, cls)
        cls.__call__ = partial(DSLMeta.__handle_batch, cls)
        cls.__init__ = partial(DSLMeta.__initialize_class, cls)
        register()(cls)
        DSLMeta.__add_to_collection(cls)

    def __initialize_class(cls, on_invalid_command: str = "Простите, я вас не понял", *args, **kwargs):
        # message to be sent on message with no associated handler
        cls.on_invalid_command = on_invalid_command

    def __handle_batch(cls: 'DSLMeta', utterances_batch: List, history_batch: List = [], states_batch: List = []) \
            -> Tuple[List, ...]:
        responses_batch, confidences_batch, new_states_batch = [], [], []
        for utterance, history, state in zip_longest(utterances_batch, history_batch, states_batch):
            response, confidence, new_state = cls.__handle(cls, utterance, history, state)
            responses_batch.append(response)
            confidences_batch.append(confidence)
            new_states_batch.append(new_state)
        return responses_batch, confidences_batch, new_states_batch

    @staticmethod
    def __add_to_collection(cls: 'DSLMeta'):
        DSLMeta.skill_collection[cls.name] = cls

    @staticmethod
    def __handle(cls: 'DSLMeta',
                 utterance: str,
                 history: str,
                 state: str) -> ResponseType:
        """
        Handles what is going to be after a message from user arrived.
        Simple usage:
        ExampleSkill.handle(<message>, <user_id>)

        :param cls: instance of callee's class
        :param utterance: a message to be handled
        :param history: history of dialog
        :param state: state
        :return: handler function's result if succeeded
        """
        current_handler = cls.__select_handler(utterance, history, state)
        return cls.__run_handler(current_handler, utterance, history, state)

    def __select_handler(cls, message: str,
                         history: str,
                         state: str) -> Optional[Callable]:
        """
        Selects handler that will process request
        :return: handler function that is selected and None if no handler fits request
        """
        available_handlers = cls.state_to_handler[state]
        available_handlers.extend(cls.universal_handlers)
        available_handlers.sort(key=lambda h: h.priority, reverse=True)
        for handler in available_handlers:
            if handler.check(message, history):
                return handler.func

    def __run_handler(cls, handler: Optional[Callable],
                      message: str,
                      history: str,
                      state: str) -> ResponseType:
        """
        Runs specified handler for current message and context
        :param handler: handler to be run. If None, on_invalid_command is returned
        :return: ResponseType
        """
        if handler is None:
            return ResponseType(cls.on_invalid_command, 0.0, None)
        try:
            return ResponseType(*handler(message, history, state))
        except Exception as exc:
            return ResponseType(str(exc), 1.0, None)

    @staticmethod
    def handler(commands: List[str] = None,
                state: str = None,
                context_condition=None,
                priority: int = 0):
        """
        Decorator to be used in skills' classes.
        Sample usage:
        class ExampleSkill(metaclass=ZDialog):
            @ZDialog.handler(commands=["hello", "hey"], state="greeting")
            def __greeting(message: str):
                ...

        :param priority: integer value to indicate priority. If multiple handlers satisfy
                          all the requirements, the handler with the greatest priority value will be used
        :param context_condition: function that takes context and
                                  returns True if this handler should be enabled
                                  and False otherwise. If None, no condition is checked
        :param commands: phrases/regexs on what the function wrapped
                         by this decorator will trigger
        :param state: state name
        :return: function decorated into Handler class
        """
        if commands is None:
            commands = [".*"]

        def decorator(func: Union[Callable, Handler]) -> Handler:
            return RegexHandler(expand_arguments(func), commands,
                                context_condition=context_condition,
                                priority=priority, state=state)

        return decorator
