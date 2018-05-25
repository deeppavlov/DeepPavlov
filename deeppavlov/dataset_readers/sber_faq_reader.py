from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done
from deeppavlov.core.commands.utils import get_deeppavlov_root, expand_path
import random
import csv

@register('sber_faq_reader')
class SberFAQReader(DatasetReader):
    
    def read(self, data_path):
        data_path = expand_path(data_path)
        self.download_data(data_path)
        dataset = {'train': None, 'valid': None, 'test': None}
        train_fname = Path(data_path) / 'sber_faq_train.csv'
        valid_fname = Path(data_path) / 'sber_faq_val.csv'
        test_fname = Path(data_path) / 'sber_faq_test.csv'
        self.sen2int_vocab = {}
        self.classes_vocab = {}
        self._build_sen2int_classes_vocabs(train_fname, valid_fname, test_fname)
        dataset["train"] = self.preprocess_data(train_fname, num_neg=1000)
        dataset["valid"] = self.preprocess_data(valid_fname)
        dataset["test"] = self.preprocess_data(test_fname)
        return dataset
    
    def download_data(self, data_path):
        pass
        if not is_done(Path(data_path)):
            download_decompress(url="http://lnsigo.mipt.ru/export/datasets/insuranceQA-master.zip",
                                download_path=data_path)
            mark_done(data_path)

    def _build_sen2int_classes_vocabs(self, train_fname, valid_fname, test_fname):
        sen = []
        label = []
        with open(train_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(el[0])
                label.append(el[1])
        with open(valid_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(el[0])
                label.append(el[1])
        with open(test_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(el[0])
                label.append(el[1])
        self.sen2int_vocab = {el[1]: el[0] for el in enumerate(sen)}
        self.classes_vocab = {el: set() for el in set(label)}
        for el in zip(label, sen):
            self.classes_vocab[el[0]].add(self.sen2int_vocab[el[1]])

    def preprocess_data(self, fname, num_neg=9):
        contexts = []
        responses = []
        positive_responses_pool = []
        negative_responses_pool = []
        sen = []
        label = []
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(el[0])
                label.append(el[1])
        for k, v in self.classes_vocab.items():
            sen_li = list(v)
            neg_resps = self._get_neg_resps(self.classes_vocab, k)
            for s1 in sen_li:
                contexts.append(s1)
                s2 = random.choice(list(v - {s1}))
                responses.append(s2)
                positive_responses_pool.append(list(v - {s1}))

                nr = random.choices(neg_resps, k=num_neg)
                negative_responses_pool.append(nr)

        data = [{"context": el[0], "response": el[1],
                "pos_pool": el[2], "neg_pool": el[3]}
                for el in zip(contexts, responses,
                positive_responses_pool, negative_responses_pool)]
        return data

    def _get_neg_resps(self, classes_vocab, label, num_neg_resps=10):
        neg_resps = []
        for k, v in classes_vocab.items():
            if k != label:
                neg_resps.append(list(v))
        neg_resps = [x for el in neg_resps for x in el]
        return neg_resps
