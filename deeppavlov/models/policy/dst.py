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
import copy
from pathlib import Path
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)


@register('simple_dst')
class SimpleDST(Component):
    def __init__(self, commands, *args, **kwargs):
        commands_path = Path(commands)
        commands_spec = importlib.util.spec_from_file_location(commands_path.stem, commands_path)
        commands = importlib.util.module_from_spec(commands_spec)
        commands_spec.loader.exec_module(commands)
        self.commands = defaultdict(list)
        self.commands.update(commands.get())
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
        if '__COMMANDS__' in state:
            cmd = copy.deepcopy(state['__COMMANDS__'])
            del state['__COMMANDS__']
        else:
            cmd = [{'command': 'DEFAULT'}]

        p = defaultdict(list)
        p.update(params)

        for c in cmd:
            print(f'Execute command {c}')
            new_state = self.commands[c['command']](c, new_state, p)

        return new_state


