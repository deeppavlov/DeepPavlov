# encoding: utf-8

from abc import ABCMeta
from collections import defaultdict
from functools import partial
from itertools import zip_longest, starmap
from typing import List, Optional, Union, Dict, Callable, Tuple

from more_itertools import unzip

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
        cls.__init__ = partial(DSLMeta.__init__class, cls)
        register()(cls)
        DSLMeta.__add_to_collection(cls)

    def __init__class(cls, on_invalid_command: str = "Простите, я вас не понял", *args, **kwargs):
        """
        Initialize Skill class
        Args:
            on_invalid_command:  message to be sent on message with no associated handler
        """
        # message to be sent on message with no associated handler
        cls.on_invalid_command = on_invalid_command

    def __handle_batch(cls: 'DSLMeta',
                       utterances_batch: List,
                       history_batch: List = None,
                       states_batch: List = None) -> Tuple[List, ...]:
        """Returns skill inference result.
        Returns batches of skill inference results, estimated confidence
        levels and up to date states corresponding to incoming utterance
        batch.
        Args:
            utterances_batch: A batch of utterances of str type.
            history_batch: A batch of list typed histories for each utterance.
            states_batch:  A batch of arbitrary typed states for
                each utterance.
        Returns:
            response: A batch of arbitrary typed skill inference results.
            confidence: A batch of float typed confidence levels for each of
                skill inference result.
            output_states_batch:  A batch of arbitrary typed states for
                each utterance.

        """
        history_batch = history_batch or []
        states_batch = states_batch or []
        return unzip(starmap(cls.handle, zip_longest(utterances_batch, history_batch, states_batch)))

    @staticmethod
    def __add_to_collection(cls: 'DSLMeta'):
        """
        Adds Skill class to Skill classes collection
        Args:
            cls: Skill class
        """
        DSLMeta.skill_collection[cls.name] = cls

    @staticmethod
    def __handle(cls: 'DSLMeta',
                 utterance: str,
                 history: str,
                 state: str) -> ResponseType:
        """
        Handles what is going to be after a message from user arrived.
        Simple usage:
        ExampleSkill.handle(Request(<message>, <user_id>))

        Args:
            cls: instance of callee's class
            utterance: a message to be handled
            history: history of dialog
            state: state
        Returns:
            result: handler function's result if succeeded
        """
        current_handler = cls.__select_handler(utterance, history, state)
        return cls.__run_handler(current_handler, utterance, history, state)

    def __select_handler(cls, message: str,
                         history: str,
                         state: str) -> Optional[Callable]:
        """
        Selects handler with the highest priority that could be triggered from the passed state and message.
        Returns:
             handler function that is selected and None if no handler fits request
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
        Args:
            handler: handler to be run. If None, on_invalid_command is returned
        Returns:
             ResponseType
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

        Args:
            priority: integer value to indicate priority. If multiple handlers satisfy
                          all the requirements, the handler with the greatest priority value will be used
            context_condition: function that takes context and
                                  returns True if this handler should be enabled
                                  and False otherwise. If None, no condition is checked
            commands: phrases/regexs on what the function wrapped
                         by this decorator will trigger
            state: state name
        Returns:
            function decorated into Handler class
        """
        if commands is None:
            commands = [".*"]

        def decorator(func: Union[Callable, Handler]) -> Handler:
            return RegexHandler(expand_arguments(func), commands,
                                context_condition=context_condition,
                                priority=priority, state=state)

        return decorator
