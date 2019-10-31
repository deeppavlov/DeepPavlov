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
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from deeppavlov.core.models.component import Component
from deeppavlov.utils.connector.dialog_logger import DialogLogger


class Agent(Component, metaclass=ABCMeta):
    """Abstract class for agents.

    Agent is an entity which receives inputs from the outer word, processes
    them and returns response to each input. Usually agent implements real-life
    task, business or user case. Agent encapsulates skills instances, management
    of skills inference and skills inference results processing. Also agent
    provides management both for history and state for each utterance and uses
    only incoming utterances IDs to distinguish them.

    Args:
        skills: List of initiated agent skills instances.

    Attributes:
        skills: List of initiated Skill or Component instances.
            Components API should should implement API of Skill abstract class.
        history: Histories for each each dialog with agent indexed
            by dialog ID. Each history is represented by list of incoming
            and outcoming replicas of the dialog casted to str and updated automatically.
        states: States for each skill with agent indexed by dialog ID. Each
            state updated automatically after each wrapped skill inference.
            So we highly recommend use this attribute only for reading and
            not to use it for your custom skills management.
        wrapped_skills: Skills wrapped to SkillWrapper objects. SkillWrapper
            object gives to Skill __call__ signature of Agent __call__ and
            handles automatic state management for skill. All skills are
            wrapped to SkillsWrapper automatically during agent initialisation.
            We highly recommend to use wrapped skills for skills inference.
        dialog_logger: DeepPavlov dialog logging facility.
    """

    def __init__(self, skills: List[Component]) -> None:
        self.skills = skills
        self.history: Dict = defaultdict(list)
        self.states: Dict = defaultdict(lambda: [None] * len(self.skills))
        self.wrapped_skills: List[SkillWrapper] = \
            [SkillWrapper(skill, skill_id, self) for skill_id, skill in enumerate(self.skills)]
        self.dialog_logger: DialogLogger = DialogLogger()

    def __call__(self, utterances_batch: list, utterances_ids: Optional[list] = None) -> list:
        """Wraps _call method and updates utterances history.

        Args:
            utterances_batch: Batch of incoming utterances.
            utterances_ids: Batch of dialog IDs corresponding to incoming utterances.

        Returns:
            responses: A batch of responses corresponding to the
                utterance batch received by agent.
        """
        responses_batch = self._call(utterances_batch, utterances_ids)

        batch_size = len(utterances_batch)
        ids = utterances_ids or list(range(batch_size))

        for utt_batch_idx, utt_id in enumerate(ids):
            self.history[utt_id].append(str(utterances_batch[utt_batch_idx]))
            self.dialog_logger.log_in(utterances_batch[utt_batch_idx], utt_id)

            self.history[utt_id].append(str(responses_batch[utt_batch_idx]))
            self.dialog_logger.log_out(responses_batch[utt_batch_idx], utt_id)

        return responses_batch

    @abstractmethod
    def _call(self, utterances_batch: list, utterances_ids: Optional[list] = None) -> list:
        """Processes batch of utterances and returns corresponding responses batch.

        Each call of Agent processes incoming utterances and returns response
        for each utterance Batch of dialog IDs can be provided, in other case
        utterances indexes in incoming batch are used as dialog IDs.

        Args:
            utterances_batch: Batch of incoming utterances.
            utterances_ids: Batch of dialog IDs corresponding to incoming utterances.

        Returns:
            responses: A batch of responses corresponding to the
                utterance batch received by agent.
        """
        pass


class SkillWrapper:
    """Skill instances wrapper for internal use in Agent.

    SkillWrapper gives to skill interface of Agent and handles automatic state
    management for skill.

    Args:
        skill: Wrapped skill.
        skill_id: Skill index in Agent.skills list.
        agent: Agent instance.

    Attributes:
        skill: Wrapped skill.
        skill_id: Skill index in Agent.skills list.
        agent: Agent instance.
    """

    def __init__(self, skill: Component, skill_id: int, agent: Agent) -> None:
        self.skill = skill
        self.skill_id = skill_id
        self.agent = agent

    def __call__(self, utterances_batch: list, utterances_ids: Optional[list] = None) -> Tuple[list, list]:
        """Wraps __call__ method of Skill instance.

            Provides skill __call__ with signature of Agent __call__ and handles
            automatic state management for skill.

        Args:
            utterances_batch: Batch of incoming utterances.
            utterances_ids: Batch of dialog IDs corresponding to incoming utterances.

        Returns:
            response: A batch of arbitrary typed skill inference results.
            confidence: A batch of float typed confidence levels for each of
                skill inference result.
            states: Optional. A batch of arbitrary typed states for each
                response.
        """
        history_batch = [self.agent.history[utt_id] for utt_id in utterances_ids]
        states_batch = [self.agent.states[utt_id][self.skill_id] for utt_id in utterances_ids]

        predicted, confidence, *states = self.skill(utterances_batch, history_batch, states_batch)

        states = states[0] if states else [None] * len(predicted)
        for utt_id, state in zip(utterances_ids, states):
            self.agent.states[utt_id][self.skill_id] = state

        return predicted, confidence
