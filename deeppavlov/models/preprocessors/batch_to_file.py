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

from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.utils import expand_path

@register('batch_to_txt_file')
class BatchToTXTFile(Component):
    """Component for saving batch of  strings to a file
    """
    def __init__(self, 
                 save_path: str,
                 valid_dir: str = 'valid',
                 train_dir: str = 'train',
                 test_dir: str = 'test',
                 file_name_prefix: str = 'prts_',
                 file_name_suffix: str = '',
                 file_start_number: int = 0,
                 *args, **kwargs):
        self.save_path = expand_path(save_path)
        self.valid_dir = valid_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.file_name_prefix = file_name_prefix
        self.file_name_suffix = file_name_suffix
        self.file_no = file_start_number
        self.tgt_path = self.save_path / self.valid_dir

    def __call__(self, batch: List[List[str]]):
        """Save batch of strings to a file

        Args:
            batch: a list containing  strings

        Returns:
            a empty list
        """
        file_path = self.tgt_path / str(self.file_name_prefix + self.file_no +  self.file_name_suffix)
        self.file_no += 1
        open(file_path, 'wt').write('\n'.join(batch))
        return []



    def process_event(self, event_name, _):
        if event_name == 'after_validation':
            self.tgt_path = self.save_path / self.train_dir
        elif event_name == 'after_epoch':
            self.tgt_path = self.save_path / self.test_dir
