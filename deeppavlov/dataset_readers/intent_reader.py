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

from logging import getLogger
from pathlib import Path

from typing import List, Any, Dict, Tuple

import re
from xeger import Xeger

from deeppavlov.dataset_readers.basic_classification_reader import BasicClassificationDatasetReader

log = getLogger(__name__)


@register("intent_reader")
class IntentReader(BasicClassificationDatasetReader):
    """
    Class provides reading datasets for IntentCatcher model.
    """

    def read(self, data_path: str, limit: int = 10, url: str = None,
             format: str = "csv", class_sep: str = None,
             *args, **kwargs) -> Dict[str, List[Tuple[str, Any]]]:
        """
        Read intent dataset from data_path directory.
        Reading files are all data_types + extension
        (i.e for data_types=["train", "valid"] files "train.csv" and "valid.csv" form
        data_path will be read).
        For "train.csv", assumed that input are regexp phrases,
        from which the actual dataset is generated.

        Args:
            data_path: directory with files
            limit: maximum number of phrases generated from regexp for train.csv
            url: download data files if data_path not exists or empty
            format: extension of files. Set of Values: ``"csv", "json"``
            class_sep: string separator of labels in column with labels
            sep (str): delimeter for ``"csv"`` files. Default: None -> only one class per sample
            header (int): row number to use as the column names
            names (array): list of column names to use
            orient (str): indication of expected JSON string format
            lines (boolean): read the file as a json object per line. Default: ``False``
        Returns:
            dictionary with types from data_types.
            Each field of dictionary is a list of tuples (x_i, y_i)
        """
        xeger = Xeger(limit=limit)
        data = super(BasicClassificationDatasetReader).read(
            data_path,
            url,
            format,
            class_sep,
            args,
            kwargs
            )
        if 'train' in data:
            generated_samples = []
            for sample in data['train']:
                text = sample[0]
                generated_texts = list({xeger.xeger(text) for _ in range(limit)})
                generated_samples.extend([(text, sample[1]) for text in generated_texts])
            data['train'] = generated_samples
        return data
