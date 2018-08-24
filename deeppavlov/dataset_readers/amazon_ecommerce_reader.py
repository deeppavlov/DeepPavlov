from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register
from deeppavlov.core.commands.utils import get_deeppavlov_root, expand_path
import re
from pathlib import Path
from typing import Dict, List, Union

@register('amazon_ecommerce_reader')
class AmazonEcommerceReader(DatasetReader):
    
    def read(self, data_path: str, catalog: str, **kwargs):
        ec_data_global = []
        data_path = Path(expand_path(data_path))

        if data_path.is_dir():
            for fname in data_path.iterdir():
                if fname.is_file():
                    if catalog in fname.name:
                        ec_data_global = self._load_amazon_ecommerce_file(fname)

        dataset = {'train': None, 'valid': None, 'test': None}
        dataset["train"] = [(item,1) for item in ec_data_global]
        dataset["valid"] = []
        dataset["test"] = []
        return dataset

    def _load_amazon_ecommerce_file(self, fname):
        ec_data = []
        item = dict()
        new_item_re = re.compile("ITEM *\d+")

        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if new_item_re.match(line):
                    if len(item.keys())>0:
                        if 'Title' in item and 'Feature' in item:
                            ec_data.append(item)
                    item = {'Item': int(line[5:]), 'Category': fname.name.split("_")[1]}
                else:
                    ro = line.strip().split("=")
                    if len(ro) == 2:
                        if ro[0] in item:
                            item[ro[0]] += "." + ro[1]
                        else:
                            item[ro[0]] = ro[1]
        return ec_data
