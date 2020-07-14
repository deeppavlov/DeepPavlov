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
import pickle
from logging import getLogger
from pathlib import Path
from typing import Dict

from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.registry import get_model, register
from deeppavlov.core.data.dataset_reader import DatasetReader


log = getLogger(__name__)


@register('multitask_reader')
class MultiTaskReader(DatasetReader):
    """Class to read several datasets simultaneuosly"""

    def read(self, data_path, tasks: Dict[str, Dict[str, str]]):
        """Creates dataset readers for tasks and returns what task dataset readers `read()` methods return.

        Args:
            data_path: can be anything since it is not used. `data_path` is present because it is
                required in train.py script.
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

        Returns:
            dictionary which keys are task names and values are what task readers `read()` methods returned.
        """
        data = {}
        for task_name, reader_params in tasks.items():
            reader_params = copy.deepcopy(reader_params)
            tasks[task_name] = from_params({"class_name": reader_params['reader_class_name']})
            del reader_params['reader_class_name']
            reader_params['data_path'] = Path(reader_params['data_path']).expanduser()
            data[task_name] = tasks[task_name].read(**reader_params)
        return data

