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

from typing import List, Optional

from deeppavlov.deprecated.agent import Agent, Filter, Processor
from deeppavlov.deprecated.agents.filters import TransparentFilter
from deeppavlov.deprecated.agents.processors import HighestConfidenceSelector
from deeppavlov.core.models.component import Component


class DefaultAgent(Agent):
    """
    DeepPavlov default implementation of Agent abstraction.

    Default Agent is an implementation of agent template, with following
    pipeline for each utterance batch received by agent:
    1) Utterance batch is processed through agent Filter which selects utterances to be processed with each agent skill;
    2) Utterances are processed through skills selected for them;
    3) Utterances and skill responses are processed through agent Processor which generates agent's response for the outer world.
    Defining DefaultAgent means:
    a) To define set of skills it uses;
    b) To implement skills Filter;
    c) To implement Processor.
    You can refer to :class:`deeppavlov.deprecated.skill.Skill`, :class:`deeppavlov.deprecated.agent.Filter`,
    :class:`deeppavlov.deprecated.agent.Processor` base classes to get more info.

    Args:
        skills: List of initiated agent skills or components instances.
        skills_processor: Initiated agent processor.
        skills_filter: Initiated agent filter.

    Attributes:
        skills: List of initiated agent skills instances.
        skills_processor: Initiated agent processor.
        skills_filter: Initiated agent filter.
    """

    def __init__(self, skills: List[Component], skills_processor: Optional[Processor] = None,
                 skills_filter: Optional[Filter] = None, *args, **kwargs) -> None:
        super(DefaultAgent, self).__init__(skills=skills)
        self.skills_filter = skills_filter or TransparentFilter(len(skills))
        self.skills_processor = skills_processor or HighestConfidenceSelector()

    def _call(self, utterances_batch: list, utterances_ids: Optional[list] = None) -> list:
        """
        Processes batch of utterances and returns corresponding responses batch.

        Each call of Agent passes incoming utterances batch through skills filter,
        agent skills, skills processor. Batch of dialog IDs can be provided, in
        other case utterances indexes in incoming batch are used as dialog IDs.

        Args:
            utterances_batch: Batch of incoming utterances.
            utterances_ids: Batch of dialog IDs corresponding to incoming utterances.

        Returns:
            responses: A batch of responses corresponding to the
                utterance batch received by agent.
        """
        batch_size = len(utterances_batch)
        ids = utterances_ids or list(range(batch_size))
        batch_history = [self.history[utt_id] for utt_id in ids]
        responses = []

        filtered = self.skills_filter(utterances_batch, batch_history)

        for skill_i, (filtered_utterances, skill) in enumerate(zip(filtered, self.wrapped_skills)):
            skill_i_utt_indexes = [utt_index for utt_index, utt_filter in enumerate(filtered_utterances) if utt_filter]

            if skill_i_utt_indexes:
                skill_i_utt_batch = [utterances_batch[i] for i in skill_i_utt_indexes]
                skill_i_utt_ids = [ids[i] for i in skill_i_utt_indexes]
                res = [(None, 0.)] * batch_size
                predicted, confidence = skill(skill_i_utt_batch, skill_i_utt_ids)

                for i, predicted, confidence in zip(skill_i_utt_indexes, predicted, confidence):
                    res[i] = (predicted, confidence)

                responses.append(res)

        responses = self.skills_processor(utterances_batch, batch_history, *responses)

        return responses
