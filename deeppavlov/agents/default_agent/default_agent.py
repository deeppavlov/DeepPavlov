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

from collections import defaultdict
from typing import List

from deeppavlov.core.agent.agent import Agent
from deeppavlov.core.agent.filter import Filter
from deeppavlov.core.agent.processor import Processor
from deeppavlov.core.skill.skill import Skill
from deeppavlov.agents.filters.transparent_filter import TransparentFilter
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector


class DefaultAgent(Agent):
    """DeepPavlov default implementation of Agent abstraction.

    Default Agent is an implementation of agent template, which following
    pipeline for each utterance batch received by agent:
        1) Utterance batch is processed through agent Filter which selects
            utterances to be processed with each agent skill;
        2) Utterances are processed through skills selected for them;
        3) Utterances and skill responses are processed through agent
            Processor which generate agent's response to the outer world.
    Defining DefaultAgent means:
        a) To define set of skills it uses;
        b) To implement skills Selector;
        c) To implement Processor.
    You can refer Skill, Processor, Selector base classes to get more info.

    Args:
        skills: List of initiated agent skills instances.
        skills_processor: Initiated agent processor.
        skills_filter: Initiated agent filter.

    Attributes:
        skills: List of initiated agent skills instances.
        skills_processor: Initiated agent processor.
        skills_filter: Initiated agent filter.
        history: Histories for each each dialog with agent indexed
            by dialog ID. Each history is represented by list of incoming
            and outcoming replicas of the dialog.
        states: States for each each dialog with agent indexed by dialog ID.
    """
    def __init__(self, skills: List[Skill], skills_processor: Processor=None,
                 skills_filter: Filter=None, *args, **kwargs):
        super(DefaultAgent, self).__init__(skills=skills)
        self.skills_filter: Filter = skills_filter or TransparentFilter(len(skills))
        self.skills_processor: Processor = skills_processor or HighestConfidenceSelector()
        self.history: dict = defaultdict(list)
        self.states: dict = defaultdict(lambda: [None] * len(self.skills))

    def _call(self, utterances_batch: list, utterances_ids: list=None) -> list:
        """Processes batch of utterances and returns corresponding responses batch.

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
        batch_states = [self.states[utt_id] for utt_id in ids]
        responses = []

        # Filter utterances to be processed with each skills
        filtered = self.skills_filter(utterances_batch, batch_history)

        for skill_i, (filtered_utterances, skill) in enumerate(zip(filtered, self.skills)):
            # get batch of indexes for utterances to which skill_i will be applied
            skill_i_utt_indexes = [utt_index for utt_index, utt_filter in enumerate(filtered_utterances) if utt_filter]

            if skill_i_utt_indexes:
                # make batches of utterances and corresponding histories and states to which skill will be applied
                batch = [[], [], []]
                for i in skill_i_utt_indexes:
                    batch[0].append(utterances_batch[i])
                    batch[1].append(batch_history[i])
                    batch[2].append(batch_states[i][skill_i])

                # make blank response vector for all utterances in incoming batch (including not processed by skill)
                res = [(None, 0.)] * batch_size

                # infer skill with utterances/histories/states batches
                predicted, confidence, *state = skill(*batch)

                # populate elements of response vector which correspond processes utterances
                state = state[0] if state else [None] * len(predicted)
                for i, predicted, confidence, state in zip(skill_i_utt_indexes, predicted, confidence, state):
                    res[i] = (predicted, confidence)

                    # update utterances/skills states
                    batch_states[i][skill_i] = state

                responses.append(res)

        responses = self.skills_processor(utterances_batch, batch_history, *responses)

        return responses
