#%%
from collections import defaultdict
from typing import List
import random

from deeppavlov.core.models.component import Component


class RandomFilter(Component):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, utterances, batch_history):
        res = []
        for _ in utterances:
            r = [bool(random.getrandbits(1)) for _ in range(self.size)]
            r[-1] = r[-1] or not any(r)
            res.append(r)
        return res


class RandomSelector(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, utterances, batch_history, *responses):
        return [random.choice([t for t, sc in r if t]) for r in zip(*responses)]


class Agent(Component):
    def __init__(self, skills: List[Component], skills_filter=None, skills_selector=None, *args, **kwargs):
        self.skills = skills
        self.skills_filter = skills_filter or (lambda u, _: [[True] * len(skills)] * len(u)) or RandomFilter(len(skills))
        self.skills_selector = skills_selector or RandomSelector()
        self.history = defaultdict(list)
        self.states = defaultdict(lambda: [None] * len(self.skills))

    def __call__(self, utterances, ids=None):
        batch_size = len(utterances)
        ids = ids or list(range(batch_size))
        batch_history = [self.history[id] for id in ids]
        batch_states = [self.states[id] for id in ids]
        filtered = self.skills_filter(utterances, batch_history)
        responses = []
        for skill_i, (m, skill) in enumerate(zip(zip(*filtered), self.skills)):
            m = [i for i, m in enumerate(m) if m]
            batch = tuple(zip(*[(utterances[i], batch_history[i], batch_states[i][skill_i]) for i in m]))
            res = [(None, 0.)] * batch_size
            if batch:
                predicted = skill(*batch)
                for i, p in zip(m, predicted):
                    state = p[-1]
                    p = p[:-1]
                    res[i] = p
                    batch_states[i][skill_i] = state
            responses.append(res)
        responses = self.skills_selector(utterances, batch_history, *responses)
        for history, utterance, response in zip(batch_history, utterances, responses):
            history.append(utterance)
            history.append(response)
        return responses


if __name__ == '__main__':
    #%%

    c1 = lambda u, b, s: [('1', 1, None) for _ in u]
    c2 = lambda u, b, s: [('2', 1, None) for _ in u]
    c3 = lambda u, b, s: [('3', 1, None) for _ in u]

    a = Agent([c1, c2, c3])

    #%%
    a(['Привет!', "Пока!", "И чо?", "Нет же"])
