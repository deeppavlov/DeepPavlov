from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done
from deeppavlov.core.commands.utils import get_deeppavlov_root, expand_path


@register('insurance_reader')
class InsuranceReader(DatasetReader):
    
    def read(self, data_path):
        data_path = expand_path(data_path)
        self.download_data(data_path)
        dataset = {'train': None, 'valid': None, 'test': None}
        train_fname = Path(data_path) / 'insurance_data/insuranceQA-master/V1/question.train.token_idx.label'
        dataset["train"] = self.preprocess_data_train(train_fname)
        valid_fname = Path(data_path) / 'insurance_data/insuranceQA-master/V1/question.dev.label.token_idx.pool'
        dataset["valid"] = self.preprocess_data_valid_test(valid_fname)
        test_fname = Path(data_path) / 'insurance_data/insuranceQA-master/V1/question.test1.label.token_idx.pool'
        dataset["test"] = self.preprocess_data_valid_test(test_fname)
        return dataset
    
    def download_data(self, data_path):
        if not is_done(Path(data_path)):
            download_decompress(url="http://lnsigo.mipt.ru/export/datasets/insuranceQA-master.zip",
                                download_path=data_path)
            mark_done(data_path)

    def preprocess_data_train(self, fname):
        positive_responses_pool = []
        contexts = []
        responses = []
        with open(fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            q, pa = eli.split('\t')
            pa_list = [int(el) - 1 for el in pa.split(' ')]
            for elj in pa_list:
                contexts.append([int(el.split('_')[1]) for el in q.split(' ')])
                responses.append(elj)
                positive_responses_pool.append(pa_list)
        train_data = [{"context": el[0], "response": el[1],
                       "pos_pool": el[2], "neg_pool": None}
                      for el in zip(contexts, responses, positive_responses_pool)]
        return train_data
    
    def preprocess_data_valid_test(self, fname):
        pos_responses_pool = []
        neg_responses_pool = []
        contexts = []
        pos_responses = []
        with open(fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            pa, q, na = eli.split('\t')
            pa_list = [int(el) - 1 for el in pa.split(' ')]
            for elj in pa_list:
                contexts.append([int(el.split('_')[1]) for el in q.split(' ')])
                pos_responses.append(elj)
                pos_responses_pool.append(pa_list)
                nas = [int(el) - 1 for el in na.split(' ')]
                nas = [el for el in nas if el not in pa_list]
                neg_responses_pool.append(nas)
        data = [{"context": el[0], "response": el[1], "pos_pool": el[2], "neg_pool": el[3]}
                for el in zip(contexts, pos_responses, pos_responses_pool, neg_responses_pool)]
        return data
