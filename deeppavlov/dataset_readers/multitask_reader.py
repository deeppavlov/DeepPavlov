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
from collections.abc import Iterable

from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.registry import get_model, register
from deeppavlov.core.data.dataset_reader import DatasetReader

log = getLogger(__name__)


@register('multitask_reader')
class MultiTaskReader(DatasetReader):
    """
    Class to read several datasets simultaneuosly
    """

    def read(self, data_path, tasks: Dict[str, Dict[str, str]] = {}, task_names=None, path=None,
             train=None, valid=None, test=None, reader_class_name=None,**kwargs):
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
            reader_class_name - name of default dataset reader
            path - parameter path for dataset reader reader_class_name
            task_names: tasks from path for which we use the params train, validation and test
            train,validation, test - parameters for dataset reader reader_class_name
        Returns:
            dictionary which keys are task names and values are what task readers `read()` methods returned.
        """
        data = {}
        if tasks is None:
            tasks = {}

        for task_name, reader_params in tasks.items():
            log.info('Processing explicitly set tasks')
            reader_params = copy.deepcopy(reader_params)
            for default_param in ['train', 'valid', 'reader_class_name']:
                # checking if parameters are defined either on the task lavel or in the function level. 
                # Note: no check for test as we can not to define it
                error_msg = f'Set {default_param} for task {task_name} or for all tasks'
                if not(default_param in reader_params or eval(default_param) is not None):
                    raise Exception(error_msg)
            param_dict = {}
            tasks[task_name] = from_params(
                {"class_name": reader_params.get('reader_class_name',reader_class_name),
                })
            reader_params = {**reader_params, **kwargs}
            if "data_path" not in reader_params:
                reader_params["data_path"] = data_path
            if 'path' not in reader_params:
                reader_params['path'] = path
            reader_params['data_path'] = Path(
                reader_params['data_path']).expanduser()
            if 'reader_class_name' in reader_params:
                del reader_params['reader_class_name']
            for param_ in ['train', 'test', 'valid']:
                if param_ not in reader_params:
                    reader_params[param_] = eval(param_)
            reader_params['name'] = task_name
            print(reader_params)
            data[task_name] = tasks[task_name].read(**reader_params)
        if task_names is not None:
            if not isinstance(task_names, Iterable):
                raise Exception(f'task_names must be iterable, but now it is {task_names}')
            log.info(
                'For all tasks set in task_names,process those that were not explicitly set')
            task_names = [k for k in task_names if k not in data]
            if valid is None:
                raise Exception('You should set valid')
            for name in task_names:
                if 'mnli' in name and '_' not in valid:
                    log.warning(
                        f'MNLI task used in default setting. Validation on MNLI-matched assumed')
                    validation_name = valid + '_matched'
                    if test is not None:
                        test_name = test+'_matched'
                else:
                    validation_name = valid
                    if test is not None:
                        test_name = test
                reader_params = {'path': path,
                                 'data_path': data_path,
                                 'name': name,
                                 'train': train,
                                 'valid': validation_name,
                                 **kwargs}
                if test is not None:
                    reader_params['test'] = test_name
                for key in reader_params:
                    if reader_params[key] is None:
                        raise Exception(f'Set value for {key} if tasks argument is None')
                if reader_class_name is None:
                    raise Exception(f'Set the argument reader_class_name if using task_names')
                tasks[name] = from_params({"class_name": reader_class_name})
                reader_params['data_path'] = Path(
                    reader_params['data_path']).expanduser()
                data[name] = tasks[name].read(**reader_params)
        return data
