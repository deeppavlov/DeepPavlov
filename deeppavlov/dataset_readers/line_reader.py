# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('line_reader')
class LineReader(DatasetReader):
    """Read txt file by lines"""

    def read(self, data_path: str = None, *args, **kwargs) -> Dict:
        """Read lines from txt file

        Args:
            data_path: path to txt file

        Returns:
            A dictionary containing training, validation and test parts of the dataset obtainable via ``train``, ``valid`` and ``test`` keys.
        """

        with open(data_path) as f:
            content = f.readlines()

        dataset = dict()
        dataset["train"] = [(line,) for line in content]
        dataset["valid"] = []
        dataset["test"] = []

        return dataset
