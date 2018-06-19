from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done
from deeppavlov.core.commands.utils import get_deeppavlov_root, expand_path
import random
import csv
import re

@register('negative_sber_faq_reader')
class SberFAQReader(DatasetReader):
    
    def read(self, data_path):
        # data_path = expand_path(data_path)
        # self.download_data(data_path)
        dataset = {'train': None, 'valid': None, 'test': None}
        train_fname = Path(data_path) / 'sber_faq_train_1849.csv'
        valid_fname = Path(data_path) / 'sber_faq_val_1849.csv'
        test_fname = Path(data_path) / 'sber_faq_test_1849.csv'
        self.sen2int_vocab = {}
        self.classes_vocab_train = {}
        self.classes_vocab_valid = {}
        self.classes_vocab_test = {}
        self._build_sen2int_classes_vocabs(train_fname, valid_fname, test_fname)
        dataset["train"] = self.preprocess_data_train()
        dataset["valid"] = self.preprocess_data_validation(valid_fname, 'valid')
        dataset["test"] = self.preprocess_data_validation(test_fname, 'test')
        return dataset
    
    def download_data(self, data_path):
        if not is_done(Path(data_path)):
            download_decompress(url="http://lnsigo.mipt.ru/export/datasets/insuranceQA-master.zip",
                                download_path=data_path)
            mark_done(data_path)

    def _build_sen2int_classes_vocabs(self, train_fname, valid_fname, test_fname):
        sen_train = []
        label_train = []
        sen_valid = []
        label_valid = []
        sen_test = []
        label_test = []

        with open(train_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen_train.append(self.clean_sen(el[0]))
                label_train.append(int(el[1]))
        with open(valid_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen_valid.append(self.clean_sen(el[0]))
                label_valid.append(int(el[1]))
        with open(test_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen_test.append(self.clean_sen(el[0]))
                label_test.append(int(el[1]))

        sen = sen_train + sen_valid + sen_test
        self.sen2int_vocab = {el[1]: el[0] for el in enumerate(sen)}

        self.classes_vocab_train = {el: set() for el in set(label_train)}
        for el in zip(label_train, sen_train):
            self.classes_vocab_train[el[0]].add(self.sen2int_vocab[el[1]])

        self.classes_vocab_valid = {el: set() for el in set(label_valid)}
        for el in zip(label_valid, sen_valid):
            self.classes_vocab_valid[el[0]].add(self.sen2int_vocab[el[1]])

        self.classes_vocab_test = {el: set() for el in set(label_test)}
        for el in zip(label_test, sen_test):
            self.classes_vocab_test[el[0]].add(self.sen2int_vocab[el[1]])

    def preprocess_data_train(self, num_neg=1000):

        classes_vocab = self.classes_vocab_train
        contexts = []
        responses = []
        positive_responses_pool = []
        negative_responses_pool = []
        labels = []
        for k, v in classes_vocab.items():
            positive_responses_pool.append(list(v))
            contexts.append(random.choice(list(v)))
            responses.append(random.choice(list(v)))
            nr = self._get_neg_resps(classes_vocab, k)
            nr = random.choices(nr, k=num_neg)
            negative_responses_pool.append(nr)
            labels.append(k)
        data = [{"context": el[0], "response": el[1],
                "pos_pool": el[2], "neg_pool": el[3], "label": el[4]}
                for el in zip(contexts, responses, positive_responses_pool, negative_responses_pool, labels)]
        return data

    def preprocess_data_validation(self, fname, type, num_neg=9):

        if type == 'valid':
            classes_vocab = self.classes_vocab_valid
        elif type == 'test':
            classes_vocab = self.classes_vocab_test

        contexts = []
        responses = []
        positive_responses_pool = []
        negative_responses_pool = []
        sen = []
        label = []
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(self.clean_sen(el[0]))
                label.append(el[1])
        for k, v in classes_vocab.items():
            sen_li = list(v)
            neg_resps = self._get_neg_resps(classes_vocab, k)
            for s1 in sen_li:
                contexts.append(s1)
                if len(list(v - {s1})) != 0:
                    s2 = random.choice(list(v - {s1}))
                else:
                    s2 = s1
                responses.append(s2)
                positive_responses_pool.append(list(v - {s1}))

                nr = random.choices(neg_resps, k=num_neg)
                negative_responses_pool.append(nr)

        data = [{"context": el[0], "response": el[1],
                "pos_pool": el[2], "neg_pool": el[3]}
                for el in zip(contexts, responses,
                positive_responses_pool, negative_responses_pool)]
        return data

    def clean_sen(self, sen):
        return re.sub('\[Клиент:.*\]', '', sen, flags=re.IGNORECASE).\
            replace('&amp, laquo, ', '').replace('&amp, raquo, ', '').\
            replace('&amp laquo ', '').replace('&amp raquo ', '').\
            replace('&amp quot ', '').replace('&amp, quot, ', '').strip()

    def _get_neg_resps(self, classes_vocab, label):
        neg_resps = []
        for k, v in classes_vocab.items():
            if k != label:
                neg_resps.append(list(v))
        neg_resps = [x for el in neg_resps for x in el]
        return neg_resps
