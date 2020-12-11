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
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download

log = getLogger(__name__)


@register("intent_dataset_reader")
class IntentDatasetReader(DatasetReader):
    """
    Class provides reading datasets for IntentCatcher model.
    """

    @overrides
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
        data_types = ["train", "valid", "test"]

        train_file = kwargs.get('train', 'train.csv')

        if not Path(data_path, train_file).exists():
            if url is None:
                raise Exception(
                    "data path {} does not exist or is empty, and download url parameter not specified!".format(
                        data_path))
            log.info("Loading train data from {} to {}".format(url, data_path))
            download(source_url=url, dest_file_path=Path(data_path, train_file))

        data = {"train": [],
                "valid": [],
                "test": []}
        for data_type in data_types:
            file_name = kwargs.get(data_type, '{}.{}'.format(data_type, format))
            if file_name is None:
                continue

            file = Path(data_path).joinpath(file_name)
            if file.exists():
                if format == 'csv':
                    keys = ('sep', 'header', 'names')
                    options = {k: kwargs[k] for k in keys if k in kwargs}
                    df = pd.read_csv(file, **options)
                elif format == 'json':
                    keys = ('orient', 'lines')
                    options = {k: kwargs[k] for k in keys if k in kwargs}
                    df = pd.read_json(file, **options)
                else:
                    raise Exception('Unsupported file format: {}'.format(format))

                x = kwargs.get("x", "text")
                y = kwargs.get('y', 'labels')
                if isinstance(x, list):
                    if class_sep is None:
                        # each sample is a tuple ("text", "label")
                        data[data_type] = [([row[x_] for x_ in x], str(row[y]))
                                           for _, row in df.iterrows()]
                    else:
                        # each sample is a tuple ("text", ["label", "label", ...])
                        data[data_type] = [([row[x_] for x_ in x], str(row[y]).split(class_sep))
                                           for _, row in df.iterrows()]
                else:
                    if class_sep is None:
                        # each sample is a tuple ("text", "label")
                        data[data_type] = [(row[x], str(row[y])) for _, row in df.iterrows()]
                    else:
                        # each sample is a tuple ("text", ["label", "label", ...])
                        data[data_type] = [(row[x], str(row[y]).split(class_sep)) for _, row in df.iterrows()]

                if data_type == 'train':
                    generated_samples = []
                    for sample in data[data_type]:
                        text = sample[0]
                        generated_texts = list({xeger.xeger(text) for _ in range(limit)})
                        generated_samples.extend([(text, sample[1]) for text in generated_texts])
                    data[data_type] = generated_samples
            else:
                log.warning("Cannot find {} file".format(file))

        return data
