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

import json
from typing import Callable, Optional

from deeppavlov import train_model
from deeppavlov.skills.dsl_skill.context import UserContext
from .handler import Handler


class FAQHandler(Handler):
    """
    FAQHandler performs matching between the user request and FAQ database using a specified matching model.

    Attributes:
        func: handler function
        model_config: DeepPavlov-compatible model config for an FAQ skill
        score_threshold: [0-1], a lower bound on the acceptable closeness of the user question to
         a question in the FAQ
        state: state in which handler can be activated
        priority: priority of the function. If 2 or more handlers can be activated, function
         with the highest priority is selected
        context_condition: predicate that accepts user context and checks if the handler should be activated.
         Example: `lambda context: context.user_id != 1` checks if user_id is not equal to 1.
         That means a user with id 1 will be always ignored by the handler.

    """

    def __init__(self,
                 func: Callable,
                 model_config: dict,
                 score_threshold: float,
                 state: Optional[str] = None,
                 context_condition: Optional[Callable] = None,
                 priority: int = 0):
        super().__init__(func, state, context_condition, priority)
        self.model_config = model_config
        self.score_threshold = score_threshold
        self.faq_model = None

    def train(self):
        """
        Trains the model
        """
        self.model_config['dataset_reader']['class_name'] = "faq_dict_reader"
        self.faq_model = train_model(self.model_config, download=True)
        return self

    def check(self, context: UserContext) -> bool:
        """
        Checks:
         - if the handler function should be triggered based on the given context via context condition.
         - if at least one of the FAQ intents is matched to the `context.message`.

        Args:
            context: user context

        Returns:
            True, if handler should be activated, False otherwise
        """
        is_previous_matches = super().check(context)
        if not is_previous_matches:
            return False

        message = ' '.join(context.message)
        results = self.faq_model([message])
        results = list(zip(*results))
        for option, score in results:
            if score > self.score_threshold:
                return True
        return False

    def expand_context(self, context: UserContext) -> UserContext:
        context.handler_payload = {'faq_options': []}

        message = ' '.join(context.message)
        results = self.faq_model([message])
        results = list(zip(*results))
        for option, score in results:
            (intent_name, intent_body) = list(json.loads(option).items())[0]
            if score > self.score_threshold:
                context.handler_payload['faq_options'].append((intent_name, intent_body, score))
        if context.handler_payload['faq_options']:
            context.handler_payload['faq_options'].sort(key=lambda x: x[2], reverse=True)
        return context
