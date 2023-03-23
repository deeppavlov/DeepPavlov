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

import copy
from logging import getLogger
from typing import Dict

from deeppavlov.core.common.registry import get_model, register
from deeppavlov.core.data.dataset_reader import DatasetReader

log = getLogger(__name__)


@register('multitask_reader')
class MultiTaskReader(DatasetReader):
    """Class to read several datasets simultaneously."""

    def read(self, tasks: Dict[str, Dict[str, dict]], task_defaults: dict = None, **kwargs):
        """Creates dataset readers for tasks and returns what task dataset readers `read()` methods return.

        Args:
            tasks: dictionary which keys are task names and values are dictionaries with param name - value pairs for
                nested dataset readers initialization. If task has key-value pair ``'use_task_defaults': False``,
                task_defaults for this task dataset reader will be ignored.
            task_defaults: default task parameters.

        Returns:
            dictionary which keys are task names and values are what task readers `read()` methods returned.
        """
        data = dict()
        if task_defaults is None:
            task_defaults = dict()
        for task_name, task_params in tasks.items():
            if task_params.pop('use_task_defaults', True) is True:
                task_config = copy.deepcopy(task_defaults)
                task_config.update(task_params)
            else:
                task_config = task_params
            reader = get_model(task_config.pop('class_name'))()
            data[task_name] = reader.read(**task_config)
        return data
