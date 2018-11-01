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

from pathlib import Path

from typing import Dict, Optional
from overrides import overrides
import glob

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)

@register('file_paths_reader')
class FilePathsReader(DatasetReader):
    """Find all file paths by a data path regex"""

    @overrides
    def read(self, data_path: str, train: Optional[str] = None,
             valid: Optional[str] = None, test: Optional[str] = None,
             *args, **kwargs) -> Dict:
        """
        Find all file paths by a data path regex

        Args:
            data_path: directory with data
            train: data path regex
            valid: data path regex
            test: data path regex

        Returns:
            dictionary with types from data_types.
            A dictionary containing training, validation and test parts of the dataset obtainable via ``train``, ``valid`` and ``test`` keys.
        """

        dataset = dict()
        dataset["train"] = self._get_files(data_path,train)
        dataset["valid"] = self._get_files(data_path,valid)
        dataset["test"] = self._get_files(data_path,test)
        return dataset

    def _get_files(self, data_path, tgt):
        if tgt is not None:
            paths = Path(data_path, tgt).absolute().as_posix()
            files = glob.glob(paths)
            files = [file for file in files if Path(file).is_file()]
            if not(files):
                raise Exception(f"Not find files. Data path '{paths}' does not exist or does not hold files!")
            else:
                log.info(f"Found {len(files)} files located '{paths}'.")
        else:
            files = []
        return files
