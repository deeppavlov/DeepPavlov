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


@register('simple_dst')
class SimpleDST(Component):
    def __init__(self, update_policy, *args, **kwargs):
        policy_path = Path(update_policy)
        policy_spec = importlib.util.spec_from_file_location(policy_path.stem, policy_path)
        self.policy = importlib.util.module_from_spec(policy_spec)
        policy_spec.loader.exec_module(self.policy)
        self.state = []

    def __call__(self, *args, **kwargs):
        logger.debug(f"State before update: {self.state}")

        batch = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                batch.append(arg)
            else:
                batch.append([arg])
        batch = zip(*batch)
        params_batch = []
        for dicts in batch:
            params = defaultdict(list)
            for d in dicts:
                params = {**params, **d}
            params_batch.append(params)

        logger.debug(f"New state params:  {params_batch}")

        result = []
        if not self.state:
            for params in params_batch:
                result.append(self._update_state(defaultdict(list), params))
        else:
            for s, ps in zip(self.state, params_batch):
                result.append(self._update_state(s, ps))

        self.state = result

        logger.debug(f"State after update: {self.state}")

        return result

    def _update_state(self, state, params):
        new_state = state
        p = defaultdict(list)
        p.update(params)
        for f in self.policy.get():
            new_state = f(new_state, p)
        return new_state


