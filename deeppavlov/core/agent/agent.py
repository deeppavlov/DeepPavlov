from collections import defaultdict
from typing import List

from deeppavlov.core.models.component import Component
from deeppavlov.core.agent.filter import Filter
from deeppavlov.core.agent.processor import Processor
from deeppavlov.core.skill.skill import Skill
from deeppavlov.agents.filters.transparent_filter import TransparentFilter
from deeppavlov.agents.processors.highest_confidence_selector import HighestConfidenceSelector


class Agent(Component):
    """Class for agent entity.

    Agent is an entity which receives inputs from outer word, processes them
    and returns response to each input. Usually agent implements real-life
    task, business or user case. To define Agent means to define a) set of
    skills it uses, b) skills selector which is used to route agent inputs
    to agent certain skills, c) skills processor which is used to process
    skills output and general agent's response for each incoming utterance.
    Agent encapsulate management both for history and state for each
    utterance and uses only utterances IDs to distinguish them.

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
        self.skills: List[Skill] = skills
        self.skills_filter: Filter = skills_filter or TransparentFilter(len(skills))
        self.skills_processor: Processor = skills_processor or HighestConfidenceSelector()
        self.history: dict = defaultdict(list)
        self.states: dict = defaultdict(lambda: [None] * len(self.skills))

    def __call__(self, utterances: list, ids: list=None) -> list:
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
        batch_size = len(utterances)
        ids = ids or list(range(batch_size))
        batch_history = [self.history[id] for id in ids]
        batch_states = [self.states[id] for id in ids]
        filtered = self.skills_filter(utterances, batch_history)
        responses = []
        for skill_i, (m, skill) in enumerate(zip(zip(*filtered), self.skills)):
            m = [i for i, m in enumerate(m) if m]
            # TODO: utterances, batch_history and batch_states batches should be lists, not tuples
            batch = tuple(zip(*[(utterances[i], batch_history[i], batch_states[i][skill_i]) for i in m]))
            res = [(None, 0.)] * batch_size
            if batch:
                predicted, confidence, *state = skill(*batch)
                state = state[0] if state else [None] * len(predicted)
                for i, predicted, confidence, state in zip(m, predicted, confidence, state):
                    res[i] = (predicted, confidence)
                    batch_states[i][skill_i] = state
            responses.append(res)
        responses = self.skills_processor(utterances, batch_history, *responses)
        for history, utterance, response in zip(batch_history, utterances, responses):
            history.append(utterance)
            history.append(response)
        return responses
