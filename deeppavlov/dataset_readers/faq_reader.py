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

from pandas import read_csv

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('faq_reader')
class FaqDatasetReader(DatasetReader):
    """Reader for FAQ dataset"""

    def read(self, data_path: str = None, data_url: str = None, x_col_name: str = 'x', y_col_name: str = 'y') -> Dict:
        """
        Read FAQ dataset from specified csv file or remote url

        Parameters:
            data_path: path to csv file of FAQ
            data_url: url to csv file of FAQ
            x_col_name: name of Question column in csv file
            y_col_name: name of Answer column in csv file

        Returns:
            A dictionary containing training, validation and test parts of the dataset obtainable via
            ``train``, ``valid`` and ``test`` keys.
        """

        if data_url is not None:
            data = read_csv(data_url)
        elif data_path is not None:
            data = read_csv(data_path)
        else:
            raise ValueError("Please specify data_path or data_url parameter")

        x = data[x_col_name]
        y = data[y_col_name]

        train_xy_tuples = [(x[i].strip(), y[i].strip()) for i in range(len(x))]

        dataset = dict()
        dataset["train"] = train_xy_tuples
        dataset["valid"] = []
        dataset["test"] = []

        return dataset
