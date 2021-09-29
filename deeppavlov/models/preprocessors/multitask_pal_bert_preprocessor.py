# Copyright 2021 Neural Networks and Deep Learning lab, MIPT
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
from typing import Iterable
from logging import getLogger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('multitask_pal_bert_preprocessor')
class MultitaskPalBertPreprocessor(Component):
    """
    Extracts out the task_id from the first index of each example for each task
    """

    def __init__(self, *args, **kwargs):
        self.n_task = len(kwargs["in"])

    def __call__(self, *args):
        out = []
        print('Preproc input')
        print(str(args))
        for task_no in range(self.n_task):
            examples = args[task_no]
            # print(examples[:5])
            task_data = []
            for values in examples:
                if isinstance(values, Iterable):
                    print(values)
                    task_id = task_no
                    if isinstance(task_id, int):
                        task_data.extend([*values[1:]])
                    else:
                        task_data.append(values)
                else:
                    pass
            if task_data:
                assert '-1,' not in str(task_data), (examples, task_data)
                out.append(tuple(task_data))
        ans = [task_id, *out]
        print('Preproc output')
        print(ans)
        # breakpoint()
        return ans
