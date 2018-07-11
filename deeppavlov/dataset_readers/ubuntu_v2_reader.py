from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done
from deeppavlov.core.commands.utils import get_deeppavlov_root, expand_path
import random
import csv
import re

@register('ubuntu_v2_reader')
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
        self._build_sen2int_classes_vocabs(train_fname, valid_fname, test_fname)
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

    def _build_sen2int_classes_vocabs(self, train_fname, valid_fname, test_fname):
        cont_train = []
        resp_train = []
        cont_valid = []
        resp_valid = []
        cont_test = []
        resp_test = []

        with open(train_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                cont_train.append(el[0])
                resp_train.append(el[1])
        with open(valid_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                cont_valid.append(el[0])
                resp_valid += el[1:]
        with open(test_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                cont_test.append(el[0])
                resp_test += el[1:]

        sen = cont_train + resp_train + cont_valid + resp_valid + cont_test + resp_test
        self.sen2int_vocab = {el[1]: el[0] for el in enumerate(sen)}

    def preprocess_data_train(self, train_fname):
        contexts = []
        responses = []
        labels = []
        with open(train_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                contexts.append(self.sen2int_vocab[el[0]])
                responses.append(self.sen2int_vocab[el[1]])
                labels.append(int(el[2]))
        data = [{"context": el[0], "response": el[1],
                "pos_pool": [el[1]], "neg_pool": None, "label": el[2]}
                for el in zip(contexts, responses, labels)]
        return data

    def preprocess_data_validation(self, fname):
        contexts = []
        responses = []
        negative_responses_pool = []
        with open(fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                contexts.append(self.sen2int_vocab[el[0]])
                responses.append(self.sen2int_vocab[el[1]])
                negative_responses_pool.append([self.sen2int_vocab[x] for x in el[2:]])
        data = [{"context": el[0], "response": el[1],
                "pos_pool": [el[1]], "neg_pool": el[2]}
                for el in zip(contexts, responses, negative_responses_pool)]
        return data