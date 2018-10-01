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
            agent skills to be applied to each utterance of the batch;
        2) Utterances are processed through skills selected for them;
        3) Utterances and skill responses are processed through agent
            Processor which generate agent's response to the outer world.
    Defining DefaultAgent means:
        a) To define set of skills it uses;
        b) To implement skills Selector;
        c) To implement Processor.
    You can refer Skill, Processor, Selector base classes to get more info.

    Args:
        skill: List of initiated agent skills instances.
        skills_processor: Initiated agent processor.
        skills_filter: Initiated agent filter.

    Attributes:
        skill: List of initiated agent skills instances.
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

    def __call__(self, utterances_batch: list, utterances_ids: list=None) -> list:
        """Processes batch of utterances and returns corresponding responses batch.

        Each call of Agent passes incoming utterances batch through skills filter,
        agent skills, skills processor. Batch of dialog IDs can be provided, in
        other case utterances indexes in incoming batch are used as dialog IDs.

        Args:
            utterances: Batch of incoming utterances.
            ids: Batch of dialog IDs corresponding to incoming utterances.

        Returns:
            responses: A batch of responses corresponding to the
                utterance batch received by agent.
        """
        batch_size = len(utterances_batch)
        ids = utterances_ids or list(range(batch_size))
        batch_history = [self.history[id] for id in ids]
        batch_states = [self.states[id] for id in ids]
        filtered = self.skills_filter(utterances_batch, batch_history)
        responses = []
        for skill_i, (m, skill) in enumerate(zip(zip(*filtered), self.skills)):
            m = [i for i, m in enumerate(m) if m]
            # TODO: utterances, batch_history and batch_states batches should be lists, not tuples
            batch = tuple(zip(*[(utterances_batch[i], batch_history[i], batch_states[i][skill_i]) for i in m]))
            res = [(None, 0.)] * batch_size
            if batch:
                predicted, confidence, *state = skill(*batch)
                state = state[0] if state else [None] * len(predicted)
                for i, predicted, confidence, state in zip(m, predicted, confidence, state):
                    res[i] = (predicted, confidence)
                    batch_states[i][skill_i] = state
            responses.append(res)
        responses = self.skills_processor(utterances_batch, batch_history, *responses)
        for history, utterance, response in zip(batch_history, utterances_batch, responses):
            history.append(utterance)
            history.append(response)
        return responses
