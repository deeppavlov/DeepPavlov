# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

from abc import ABCMeta, abstractmethod
from typing import List
from collections import defaultdict

from deeppavlov.core.models.component import Component
from deeppavlov.core.skill.skill import Skill


class Agent(Component, metaclass=ABCMeta):
    """Abstract class for agents.

    Agent is an entity which receives inputs from the outer word, processes
    them and returns response to each input. Usually agent implements real-life
    task, business or user case. Agent encapsulates skills instances, management
    of skills inference and skills inference results processing. Also agent
    provides management both for history and state for each utterance and uses
    only incoming utterances IDs to distinguish them.

    Args:
        skill: List of initiated agent skills instances.

    Attributes:
        skill: List of initiated agent skills instances.
    """
    def __init__(self, skills: List[Skill]):
        self.skills: List[Skill] = skills
        self.history: dict = defaultdict(list)
        self.states: dict = defaultdict(lambda: [None] * len(self.skills))

    def __call__(self, utterances_batch: list, utterances_ids: list=None) -> list:
        """Processes batch of utterances and returns corresponding responses batch.

        Each call of Agent processes incoming utterances and returns response
        for each utterance Batch of dialog IDs can be provided, in other case
        utterances indexes in incoming batch are used as dialog IDs.

        Args:
            utterances: Batch of incoming utterances.
            ids: Batch of dialog IDs corresponding to incoming utterances.

        Returns:
            responses: A batch of responses corresponding to the
                utterance batch received by agent.
        """
        responses_batch = self._call(utterances_batch, utterances_ids)

        batch_size = len(utterances_batch)
        ids = utterances_ids or list(range(batch_size))

        for utt_batch_idx, utt_id in enumerate(ids):
            self.history[utt_id].append(utterances_batch[utt_batch_idx])
            self.history[utt_id].append(responses_batch[utt_batch_idx])

        return responses_batch

    @abstractmethod
    def _call(self, utterances_batch: list, utterances_ids: list=None) -> list:
        """Processes batch of utterances and returns corresponding responses batch.

        Each call of Agent processes incoming utterances and returns response
        for each utterance Batch of dialog IDs can be provided, in other case
        utterances indexes in incoming batch are used as dialog IDs.

        Args:
            utterances: Batch of incoming utterances.
            ids: Batch of dialog IDs corresponding to incoming utterances.

        Returns:
            responses: A batch of responses corresponding to the
                utterance batch received by agent.
        """
