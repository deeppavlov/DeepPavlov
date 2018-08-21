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
    def __init__(self, policy, *args, **kwargs):
        policy_path = Path(policy)
        policy_spec = importlib.util.spec_from_file_location(policy_path.stem, policy_path)
        p = importlib.util.module_from_spec(policy_spec)
        policy_spec.loader.exec_module(p)
        self.policy = p.get()

        self.dst = kwargs['dst']

    def __call__(self, state, *args, **kwargs):
        result = []
        print(state)
        for s in state:
            result.append(self._perform_action(s))
        print(result)
        response, state = zip(*result)
        if isinstance(state, tuple):
            state = [state[0]]
        self.dst.state = state
        logger.debug(f"Final state: {self.dst.state}")
        return response

    def _perform_action(self, state):
        s = defaultdict(list)
        s.update(state)
        for condition, action in self.policy:
            if condition(s):
                response, state = action(s)
                if response is not None:
                    return response, state


