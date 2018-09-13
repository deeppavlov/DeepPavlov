from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done
from deeppavlov.core.commands.utils import get_deeppavlov_root, expand_path
import random
import csv
import re

@register('ubuntu_v2_reader_mt')
class UbuntuV2Reader(DatasetReader):
    
    def read(self, data_path):
        # data_path = expand_path(data_path)
        # self.download_data(data_path)
        dataset = {'train': None, 'valid': None, 'test': None}
        train_fname = Path(data_path) / 'train.csv'
        valid_fname = Path(data_path) / 'valid.csv'
        test_fname = Path(data_path) / 'test.csv'
        self.sen2int_vocab = {}
        self.classes_vocab_train = {}
        self.classes_vocab_valid = {}
        self.classes_vocab_test = {}
        dataset["train"] = self.preprocess_data_train(train_fname)
        dataset["valid"] = self.preprocess_data_validation(valid_fname)
        dataset["test"] = self.preprocess_data_validation(test_fname)
        return dataset
    
    def download_data(self, data_path):
        # if not is_done(Path(data_path)):
        #     download_decompress(url="http://lnsigo.mipt.ru/export/datasets/insuranceQA-master.zip",
        #                         download_path=data_path)
        #     mark_done(data_path)
        pass

    def preprocess_data_train(self, train_fname):
        contexts = []
        responses = []
        labels = []
        with open(train_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                contexts.append(el[0])
                responses.append(el[1])
                labels.append(int(el[2]))
        data = list(zip(contexts, responses))
        data = list(zip(data, labels))
        return data

    def preprocess_data_validation(self, fname):
        contexts = []
        responses = []
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                contexts.append(el[0])
                responses.append(el[1:])
        data = [[el[0]] + el[1] for el in zip(contexts, responses)]
        data = [(el, 1) for el in data]
        return data