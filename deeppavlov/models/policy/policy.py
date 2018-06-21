"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



from typing import List, Generator, Any
from collections import defaultdict
import importlib.util
from pathlib import Path
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)


@register('simple_policy')
class SimplePolicy(Component):
    def __init__(self, policy, nlg, *args, **kwargs):
        print(Path('.').resolve())
        policy_path = Path(policy)
        nlg_path = Path(nlg)
        policy_spec = importlib.util.spec_from_file_location(policy_path.stem, policy_path)
        nlg_spec = importlib.util.spec_from_file_location(nlg_path.stem, nlg_path)
        self.policy = importlib.util.module_from_spec(policy_spec)
        self.nlg = importlib.util.module_from_spec(nlg_spec)
        policy_spec.loader.exec_module(self.policy)
        nlg_spec.loader.exec_module(self.nlg)

    def __call__(self, *args, **kwargs):
        batch = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                batch.append(arg)
            else:
                batch.append([arg])

        batch = zip(*batch)
        new_batch = []
        for sample in batch:
            new_sample = defaultdict(list)
            for param in sample:
                new_sample = {**new_sample, **param}
            new_batch.append(new_sample)

        return [self._generate_response(*self._choose_action(b)) for b in new_batch]

    def _choose_action(self, slots):
        s = defaultdict(list)
        s.update(slots)
        for condition, action in self.policy.get():
            if condition(s):
                a, p = action(s)
                if a is not None:
                    return (a, p), s

    def _generate_response(self, action, state):
        return [self.nlg.get(*action), state]
