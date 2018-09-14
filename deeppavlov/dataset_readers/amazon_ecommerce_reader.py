# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from pathlib import Path
from typing import List, Any, Dict, Iterable, Tuple

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done

@register('amazon_ecommerce_reader')
class AmazonEcommerceReader(DatasetReader):
    """Class to download and load ecommerce data catalog"""

    def read(self, data_path: str, catalog: str, **kwargs) -> Dict[str, List[Tuple[Any, Any]]]:
        """Load data from specific catalog

        Parameters:
            data_path: path where to store the dataset
            catalog: name of the specific part of the catalog

        Returns:
            dataset: loaded dataset
        """

        ec_data_global: List[Any] = []
        data_path = Path(expand_path(data_path))

        if not is_done(data_path):
            self._download_data(data_path)

        if data_path.is_dir():
            for fname in data_path.rglob("*.txt"):
                if catalog in fname.name:
                    ec_data_global += self._load_amazon_ecommerce_file(fname)

        dataset = {
            'train': [((item, {}), ) for item in ec_data_global],
            'valid': [],
            'test':  []
            }

        return dataset

    def _download_data(self, data_path: str) -> None:
        """Download dataset"""
        url = "https://github.com/SamTube405/Amazon-E-commerce-Data-set/archive/master.zip"
        download_decompress(url, data_path)
        mark_done(data_path)

    def _load_amazon_ecommerce_file(self, fname: str) -> List[Dict[Any, Any]]:
        """Parse dataset

        Parameters:
            fname: catalog file
            
        Returns:
            ec_data: parsed catalog data
        """

        ec_data = []
        item: Dict = {}
        new_item_re = re.compile("ITEM *\d+")

        with open(fname, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                if new_item_re.match(line):
                    if len(item.keys()) > 0:
                        if 'Title' in item and 'Feature' in item:
                            ec_data.append(item)
                    item = {'Item': int(line[5:]), 'Category': fname.name.split("_")[1]}
                else:
                    row = line.strip().split("=")
                    if len(row) == 2:
                        if row[0] in item:
                            item[row[0]] += "." + row[1]
                        else:
                            item[row[0]] = row[1]
        return ec_data
