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

from typing import List, Dict, Tuple, Any


class DatasetReader:
    """An abstract class for reading data from some location and construction of a dataset."""

    def read(self, data_path: str, *args, **kwargs) -> Dict[str, List[Tuple[Any, Any]]]:
        """Reads a file from a path and returns data as a list of tuples of inputs and correct outputs
         for every data type in ``train``, ``valid`` and ``test``.
        """
        raise NotImplementedError
