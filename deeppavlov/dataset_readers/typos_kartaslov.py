import csv
import os
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import is_done, download, mark_done
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('typos_kartaslov_reader')
class TyposKartaslov(DatasetReader):
    def __init__(self):
        pass

    @staticmethod
    def build(data_path: str):
        data_path = os.path.join(data_path, 'kartaslov')

        fname = 'orfo_and_typos.L1_5.csv'
        fname = os.path.join(data_path, fname)

        if not is_done(data_path):
            url = 'https://raw.githubusercontent.com/dkulagin/kartaslov/master/dataset/orfo_and_typos/orfo_and_typos.L1_5.csv'

            download(fname, url)

            mark_done(data_path)

            print('Built')
        return fname

    @staticmethod
    def read(data_path: str, *args, **kwargs):
        fname = TyposKartaslov.build(data_path)
        with open(fname, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader)
            res = [(mistake, correct) for correct, mistake, weight in reader]
        return {'train': res}

    @overrides
    def save_vocab(self, *args, **kwargs):
        pass
