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

from abc import ABCMeta
from collections import defaultdict
from functools import partial
from itertools import zip_longest, starmap
from typing import List, Optional, Dict, Callable, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.skills.dsl_skill.context import UserContext
from deeppavlov.skills.dsl_skill.handlers.handler import Handler
from deeppavlov.skills.dsl_skill.handlers.regex_handler import RegexHandler
from deeppavlov.skills.dsl_skill.utils import SkillResponse, UserId


class DSLMeta(ABCMeta):
    """
    This metaclass is used for creating a skill. Skill is register by its class name in registry.

    Example:

    .. code:: python

            class ExampleSkill(metaclass=DSLMeta):
                @DSLMeta.handler(commands=["hello", "hey"])
                def __greeting(context: UserContext):
                    response = "Hello, my friend!"
                    confidence = 1.0
                    return response, confidence

    Attributes:
        name: class name
        state_to_handler: dict with states as keys and lists of Handler objects as values
        user_to_context: dict with user ids as keys and UserContext objects as values
        universal_handlers: list of handlers that can be activated from any state

    """
    skill_collection: Dict[str, 'DSLMeta'] = {}

    def __init__(cls, name: str,
                 bases,
                 namespace,
                 **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls.name = name
        cls.state_to_handler = defaultdict(list)
        cls.user_to_context = defaultdict(UserContext)
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

    def __init__class(cls,
                      on_invalid_command: str = "Простите, я вас не понял",
                      null_confidence: float = 0,
                      *args, **kwargs) -> None:
        """
        Initialize Skill class

        Args:
            on_invalid_command: message to be sent on message with no associated handler
            null_confidence: the confidence when DSL has no handler that fits request
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
    def __add_to_collection(cls: 'DSLMeta') -> None:
        """
        Adds Skill class to Skill classes collection

        Args:
            cls: Skill class

        """
        DSLMeta.skill_collection[cls.name] = cls

    @staticmethod
    def __handle(cls: 'DSLMeta',
                 utterance: str,
                 user_id: UserId) -> SkillResponse:
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

        current_handler = cls.__select_handler(context)
        return cls.__run_handler(current_handler, context)

    def __select_handler(cls,
                         context: UserContext) -> Optional[Callable]:
        """
        Selects handler with the highest priority that could be triggered from the passed context.

        Returns:
             handler function that is selected and None if no handler fits request

        """
        available_handlers = cls.state_to_handler[context.current_state]
        available_handlers.extend(cls.universal_handlers)
        available_handlers.sort(key=lambda h: h.priority, reverse=True)
        for handler in available_handlers:
            if handler.check(context):
                handler.expand_context(context)
                return handler.func

    def __run_handler(cls, handler: Optional[Callable],
                      context: UserContext) -> SkillResponse:
        """
        Runs specified handler for current context

        Args:
            handler: handler to be run. If None, on_invalid_command is returned
            context: user context

        Returns:
             SkillResponse

        """
        if handler is None:
            return SkillResponse(cls.on_invalid_command, cls.null_confidence)
        try:
            return SkillResponse(*handler(context=context))
        except Exception as exc:
            return SkillResponse(str(exc), 1.0)

    @staticmethod
    def handler(commands: Optional[List[str]] = None,
                state: Optional[str] = None,
                context_condition: Optional[Callable] = None,
                priority: int = 0) -> Callable:
        """
        Decorator to be used in skills' classes.
        Sample usage:

        .. code:: python

            class ExampleSkill(metaclass=DSLMeta):
                @DSLMeta.handler(commands=["hello", "hi", "sup", "greetings"])
                def __greeting(context: UserContext):
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
            return RegexHandler(func, commands,
                                context_condition=context_condition,
                                priority=priority, state=state)

        return decorator
