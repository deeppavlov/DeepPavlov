# encoding: utf-8

from abc import ABCMeta
from collections import defaultdict
from functools import partial
from itertools import zip_longest, starmap
from typing import List, Optional, Dict, Callable, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.skills.dsl_skill.context import UserContext
from deeppavlov.skills.dsl_skill.handlers import Handler, RegexHandler
from deeppavlov.skills.dsl_skill.utils import expand_arguments, ResponseType, UserId


class DSLMeta(ABCMeta):
    """
    This metaclass is used for creating a skill. Skill is register by its class name in registry.

    Example:

    .. code:: python

            class ExampleSkill(metaclass=DSLMeta):
                @DSLMeta.handler(commands=["hello", "hey"])
                def __greeting(message: str):
                    response = "Hello, my friend!"
                    confidence = 1.0
                    return response, confidence
    """
    skill_collection: Dict[str, 'DSLMeta'] = {}

    def __init__(cls, name: str,
                 bases,
                 namespace,
                 **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls.name = name

        # Attribute cls.state_to_handler is dict with states as keys and lists of Handler objects as values
        cls.state_to_handler = defaultdict(list)
        # Attribute cls.user_to_context is dict with user ids as keys and UserContext objects as values
        cls.user_to_context = defaultdict(lambda: UserContext())
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

    def __init__class(cls, on_invalid_command: str = "Простите, я вас не понял",
                      null_confidence: float = 0,
                      *args, **kwargs):
        """
        Initialize Skill class
        Args:
            on_invalid_command:  message to be sent on message with no associated handler
        """
        # message to be sent on message with no associated handler
        cls.on_invalid_command = on_invalid_command
        cls.null_confidence = null_confidence

    def __handle_batch(cls: 'DSLMeta',
                       utterances_batch: List[str],
                       user_ids_batch: List[UserId]) -> Tuple[List, ...]:
        """Returns skill inference result.
        Returns batches of skill inference results, estimated confidence
        levels and up to date states corresponding to incoming utterance
        batch.
        Args:
            utterances_batch: A batch of utterances of str type.
            user_ids_batch: A batch of user ids.
        Returns:
            response_batch: A batch of arbitrary typed skill inference results.
            confidence_batch: A batch of float typed confidence levels for each of
                skill inference result.

        """
        return (*map(list, zip(*starmap(cls.handle, zip_longest(utterances_batch, user_ids_batch)))),)

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
                 user_id: UserId) -> ResponseType:
        """
        Handles what is going to be after a message from user arrived.
        Simple usage:
        skill([<message>], [<user_id>])

        Args:
            cls: instance of callee's class
            utterance: a message to be handled
            user_id: id of a user
        Returns:
            result: handler function's result if succeeded
        """
        context = cls.user_to_context[user_id]

        context.user_id = user_id
        context.message = utterance

        current_handler = cls.__select_handler(utterance, context)
        return cls.__run_handler(current_handler, utterance, context)

    def __select_handler(cls, message, context) -> Optional[Callable]:
        """
        Selects handler with the highest priority that could be triggered from the passed message and context.
        Returns:
             handler function that is selected and None if no handler fits request
        """
        available_handlers = cls.state_to_handler[context.current_state]
        available_handlers.extend(cls.universal_handlers)
        available_handlers.sort(key=lambda h: h.priority, reverse=True)
        for handler in available_handlers:
            if handler.check(message, context):
                return handler.func

    def __run_handler(cls, handler: Optional[Callable],
                      message: str,
                      context: UserContext) -> ResponseType:
        """
        Runs specified handler for current message and context
        Args:
            handler: handler to be run. If None, on_invalid_command is returned
        Returns:
             ResponseType
        """

        if handler is None:
            return ResponseType(cls.on_invalid_command, cls.null_confidence)
        try:
            return ResponseType(*handler(message, context))
        except Exception as exc:
            return ResponseType(str(exc), 1.0)

    @staticmethod
    def handler(commands: Optional[List[str]] = None,
                state: Optional[str] = None,
                context_condition: Optional[Callable] = None,
                priority: int = 0):
        """
        Decorator to be used in skills' classes.

        Sample usage:

        .. code:: python

            class ExampleSkill(metaclass=ZDialog):
                @ZDialog.handler(commands=["hello", "hey"], state="greeting")
                def __greeting(message: str):
                    response = "Hello, my friend!"
                    confidence = 1.0
                    return response, confidence

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

        def decorator(func: Callable) -> Handler:
            return RegexHandler(expand_arguments(func), commands,
                                context_condition=context_condition,
                                priority=priority, state=state)

        return decorator
