from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

import numpy as np


@register('ranking_iterator')
class RankingIterator(DataLearningIterator):

    def __init__(self, data, len_vocab,
                 sample_candidates, sample_candidates_valid, sample_candidates_test,
                 num_negative_samples, num_ranking_samples_valid, num_ranking_samples_test,
                 shuffle=False, seed=None):
        self.len_vocab = len_vocab
        self.sample_candidates = sample_candidates
        self.sample_candidates_valid = sample_candidates_valid
        self.sample_candidates_test = sample_candidates_test
        self.num_negative_samples = num_negative_samples
        self.num_ranking_samples_valid = num_ranking_samples_valid
        self.num_ranking_samples_test = num_ranking_samples_test

        np.random.seed(seed)
        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

        super().__init__(self.data, seed=seed, shuffle=shuffle)


    def gen_batches(self, batch_size, data_type="train", shuffle=True):
        y = batch_size * [np.ones(2)]
        data = self.data[data_type]
        num_steps = len(data) // batch_size
        if data_type == "train":
            if shuffle:
                np.random.shuffle(data)
            for i in range(num_steps):
                context_response_data = data[i * batch_size:(i + 1) * batch_size]
                context = [el["context"] for el in context_response_data]
                response = [el["response"] for el in context_response_data]
                negative_response = self.create_neg_resp_rand(context_response_data, batch_size, data_type)
                x = [[context[i], [response[i]]+[negative_response[i]]] for i in range(len(context_response_data))]
                yield (x, y)
        if data_type in ["valid", "test"]:
            for i in range(num_steps + 1):
                if i < num_steps:
                    context_response_data = data[i * batch_size:(i + 1) * batch_size]
                else:
                    context_response_data = data[i * batch_size:len(data)]
                context = [el["context"] for el in context_response_data]
                response_data, y = self.create_rank_resp(context_response_data, data_type)
                x = [[context[i], response_data[i]] for i in range(len(context_response_data))]
                yield (x, y)

    def create_neg_resp_rand(self, context_response_data, batch_size, data_type):
        if data_type == "train":
            sample_candidates = self.sample_candidates
        elif data_type == "valid":
            sample_candidates = self.sample_candidates_valid
        if sample_candidates == "pool":
            candidate_lists = [el["neg_pool"] for el in context_response_data]
            candidate_indices = [np.random.randint(0, np.min([len(candidate_lists[i]),
                                 self.num_negative_samples]), 1)[0]
                                 for i in range(batch_size)]
            negative_response_data = [candidate_lists[i][candidate_indices[i]] for i in range(batch_size)]
        elif sample_candidates == "global":
            candidates = []
            for i in range(batch_size):
                candidate = np.random.randint(0, self.len_vocab, 1)[0]
                while candidate in context_response_data[i]["pos_pool"]:
                    candidate = np.random.randint(0, self.len_vocab, 1)[0]
                candidates.append(candidate)
            negative_response_data = candidates
        return negative_response_data

    def create_rank_resp(self, context_response_data, data_type="valid"):
        if data_type == "valid":
            ranking_length = self.num_ranking_samples_valid
            sample_candidates = self.sample_candidates_valid
        elif data_type == "test":
            ranking_length = self.num_ranking_samples_test
            sample_candidates = self.sample_candidates_test
        if sample_candidates == "pool":
            y = [len(el["pos_pool"]) * np.ones(ranking_length) for el in context_response_data]
            response_data = []
            for i in range(len(context_response_data)):
                pos_pool = context_response_data[i]["pos_pool"]
                resp = context_response_data[i]["response"]
                pos_pool.insert(0, pos_pool.pop(pos_pool.index(resp)))
                neg_pool = context_response_data[i]["neg_pool"]
                response = pos_pool + neg_pool
                response_data.append(response[:ranking_length])

        elif sample_candidates == "global" or sample_candidates is None:
            ranking_length = self.len_vocab
            y = [len(el["pos_pool"]) * np.ones(ranking_length) for el in context_response_data]
            response_data = []
            for i in range(len(context_response_data)):
                pos_pool = context_response_data[i]["pos_pool"]
                resp = context_response_data[i]["response"]
                pos_pool.insert(0, pos_pool.pop(pos_pool.index(resp)))
                neg_pool = context_response_data[i]["neg_pool"]
                response = pos_pool + neg_pool
                response_data.append(response[:ranking_length])
        return response_data, y
