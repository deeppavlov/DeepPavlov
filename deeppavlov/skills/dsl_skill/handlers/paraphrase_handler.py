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

from typing import List, Optional, Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .utils import get_vectorizer
from .handler import Handler, UserContext


class ParaphraseHandler(Handler):
    """
    ParaphraseHandler performs matching between the user request and the list of synonyms
    using a specified matching model. If the request is similar to any of them — the handler is triggered.

    Args:
        func: function that generates a :class:`zdialog.core.Response` given a message and an optional
        :class:`zdialog.core.UserContext`.
         phrases: list of str, synonyms that should be matched by the handler
        similarity_threshold: float, [0-1], a lower bound on the acceptable closeness of the user message to
         a phrase from the list

    """
    def __init__(self,
                 func: Callable,
                 phrases: List[str],
                 similarity_threshold: float,
                 state: Optional[str] = None,
                 context_condition: Optional[Callable] = None,
                 priority: int = 0):
        super().__init__(func, state, context_condition, priority)
        self.phrases = phrases
        self.vectorizer = get_vectorizer()
        self.vectors = None
        self.train()
        self.similarity_threshold = similarity_threshold

    def check(self, context: UserContext) -> bool:
        """
        Checks:
         - if the handler function should be triggered based on the given context via context condition.
         - if at least one of the phrases is matched to the `context.message`.

        Args:
            context: user context

        Returns:
            True, if handler should be activated, False otherwise

        """
        is_previous_matches = super().check(context)
        if not is_previous_matches:
            return False

        message = ' '.join(context.message)
        message_vector = self.vectorizer([message])
        if np.any(cosine_similarity(message_vector, self.vectors) > self.similarity_threshold):
            return True
        return False

    def train(self):
        """
        Calculates and stores phrases vectors

        """
        self.vectors = self.vectorizer(self.phrases)

    def add_paraphrase(self, phrase: str):
        """
        Adds `phrase` vector to handler vectors list then fine tunes threshold

        Args:
             phrase: phrase

        """
        self.phrases.append(phrase)
        self.vectors.append(self.vectorizer([phrase])[0])

    def remove_paraphrase(self, phrase: str):
        """
        Removes `phrase` vector from handler vectors list then fine tunes threshold

        Args:
             phrase: phrase

        """
        index = self.phrases.index(phrase)
        self.vectors.pop(index)
        self.phrases.pop(index)
