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
            tasks: dictionary which keys are task names and values are dictionaries with `DatasetReader`
                subclasses specs. `DatasetReader` specs are provided in the same format as "dataset_reader"
                in the model config except for "class_name" field which has to be named "reader_class_name".
                ```json
                "tasks": {
                  "query_prediction": {
                    "reader_class_name": "basic_classification_reader",
                    "x": "Question",
                    "y": "Class",
                    "data_path": "{DOWNLOADS_PATH}/query_prediction"
                  }
                }
                ```
            reader_class_name - name of default dataset reader
            path - parameter path for dataset reader reader_class_name. Must be provided!
            task_names: tasks from path for which we use the params train, validation and test
            train,validation, test - parameters for dataset reader reader_class_name
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
