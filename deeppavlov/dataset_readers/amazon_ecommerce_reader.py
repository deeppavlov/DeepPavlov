import re
from pathlib import Path

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done

@register('amazon_ecommerce_reader')
class AmazonEcommerceReader(DatasetReader):
    
    def read(self, data_path: str, catalog: str, **kwargs):
        ec_data_global = []
        data_path = Path(expand_path(data_path))

        if not is_done(data_path):
            self._download_data(data_path)

        if data_path.is_dir():
            for fname in data_path.rglob("*.txt"):
                if catalog in fname.name:
                    ec_data_global = self._load_amazon_ecommerce_file(fname)

        dataset = {'train': None, 'valid': None, 'test': None}
        dataset["train"] = [((item, {}), ) for item in ec_data_global]
        dataset["valid"] = []
        dataset["test"] = []
        return dataset

    def _download_data(self, data_path):
        url = "https://github.com/SamTube405/Amazon-E-commerce-Data-set/archive/master.zip"
        download_decompress(url, data_path)
        mark_done(data_path)

    def _load_amazon_ecommerce_file(self, fname):
        ec_data = []
        item = dict()
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
