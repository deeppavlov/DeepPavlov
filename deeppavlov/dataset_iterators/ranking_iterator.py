from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

import numpy as np
import random
from typing import Dict, List, Tuple


@register('ranking_iterator')
class RankingIterator(DataLearningIterator):
    """The class contains methods for iterating over a dataset for ranking in training, validation and test mode.

    Note:
        Each sample in ``data['train']`` is arranged as follows:
        ``{'context': 21507, 'response': 7009, 'pos_pool': [7009, 7010], 'neg_pool': None}``.
        The context has a 'context' key in the data sample.
        It is represented by a single integer.
        The correct response has the 'response' key in the sample,
        its value is  also always a single integer.
        The list of possible correct responses (there may be several) can be
        obtained
        with the 'pos\_pool' key.
        The value of the 'response' should be equal to the one item from the
        list
        obtained using the 'pos\_pool' key.
        The list of possible negative responses (there can be a lot of them,
        100–10000) is represented by the key 'neg\_pool'.
        Its value is None, when global sampling is used, or the list of fixed
        length, when sampling from predefined negative responses is used.
        It is important that values in 'pos\_pool' and 'negative\_pool' do
        not overlap.
        Single items in 'context', 'response', 'pos\_pool', 'neg\_pool' are
        represented
        by single integers that give lists of integers
        using some dictionary `integer–list of integers`.
        These lists of integers are converted to lists of tokens with
        some dictionary `integer–token`.
        Samples in ``data['valid']`` and ``data['test']`` representation are almost the same
        as the train sample shown above.

    Args:
        data: A dictionary containing training, validation and test parts of the dataset obtainable via
            ``train``, ``valid`` and ``test`` keys.
        sample_candidates_pool: Whether to sample candidates from  a predefined pool of candidates
            for each sample in training mode. If ``False``, negative sampling from the whole data will be performed.
        sample_candidates_pool_valid: Whether to validate a model on a predefined pool of candidates for each sample.
            If ``False``, sampling from the whole data will be performed for validation.
        sample_candidates_pool_test: Whether to test a model on a predefined pool of candidates for each sample.
            If ``False``, sampling from the whole data will be performed for test.
        num_negative_samples: A size of a predefined pool of candidates
            or a size of data subsample from the whole data in training mode.
        num_ranking_samples_valid: A size of a predefined pool of candidates
            or a size of data subsample from the whole data in validation mode.
        num_ranking_samples_test: A size of a predefined pool of candidates
            or a size of data subsample from the whole data in test mode.
        seed: Random seed.
        shuffle: Whether to shuffle data.
        len_vocab: A length of a vocabulary to perform sampling in training, validation and test mode.
        pos_pool_sample: Whether to sample response from `pos_pool` each time when the batch is generated.
            If ``False``, the response from `response` will be used.
        pos_pool_rank: Whether to count samples from the whole `pos_pool` as correct answers in test / validation mode.
        random_batches: Whether to choose batches randomly or iterate over data sequentally in training mode.
        batches_per_epoch: A number of batches to choose per each epoch in training mode.
            Only required if ``random_batches`` is set to ``True``.
        triplet_mode: Whether to use a model with triplet loss.
            If ``False``, a model with crossentropy loss will be used.
        hard_triplets_sampling: Whether to use hard triplets method of sampling in training mode.
        num_positive_samples: A number of contexts to choose from `pos_pool` for each `context`.
            Only required if ``hard_triplets_sampling`` is set to ``True``.
    """

    def __init__(self,
                 data: Dict[str, List],
                 sample_candidates_pool: bool = False,
                 sample_candidates_pool_valid: bool = True,
                 sample_candidates_pool_test: bool = True,
                 num_negative_samples: int = 10,
                 num_ranking_samples_valid: int = 10,
                 num_ranking_samples_test: int = 10,
                 seed: int = None,
                 shuffle: bool = False,
                 len_vocab: int = 0,
                 pos_pool_sample: bool = False,
                 pos_pool_rank: bool = True,
                 random_batches: bool = False,
                 batches_per_epoch: int = None,
                 triplet_mode: bool = True,
                 hard_triplets_sampling: bool = False,
                 num_positive_samples: int = 5):

        self.sample_candidates_pool = sample_candidates_pool
        self.sample_candidates_pool_valid = sample_candidates_pool_valid
        self.sample_candidates_pool_test = sample_candidates_pool_test
        self.num_negative_samples = num_negative_samples
        self.num_ranking_samples_valid = num_ranking_samples_valid
        self.num_ranking_samples_test = num_ranking_samples_test
        self.len_vocab = len_vocab
        self.pos_pool_sample = pos_pool_sample
        self.pos_pool_rank = pos_pool_rank
        self.random_batches = random_batches
        self.batches_per_epoch = batches_per_epoch
        self.triplet_mode = triplet_mode
        self.hard_triplets_sampling = hard_triplets_sampling
        self.num_positive_samples = num_positive_samples

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


    def gen_batches(self, batch_size: int, data_type: str = "train", shuffle: bool = True)->\
            Tuple[List[List[Tuple[int, int]]], List[int]]:
        """Generate batches of inputs and expected outputs to train neural networks.

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching

        Returns:
            A tuple of a batch of inputs and a batch of expected outputs.

            Inputs and expected outputs have different structure and meaning
            depending on class attributes values and ``data_type``.
        """
        data = self.data[data_type]
        if self.random_batches and self.batches_per_epoch is not None and data_type == "train":
            num_steps = self.batches_per_epoch
            assert(batch_size <= len(data))
        else:
            num_steps = len(data) // batch_size
        if data_type == "train":
            if shuffle:
                np.random.shuffle(data)
            for i in range(num_steps):
                if self.random_batches:
                    context_response_data = np.random.choice(data, size=batch_size, replace=False)
                else:
                    context_response_data = data[i * batch_size:(i + 1) * batch_size]
                context = [el["context"] for el in context_response_data]
                if self.pos_pool_sample:
                    response = [random.choice(el["pos_pool"]) for el in context_response_data]
                else:
                    response = [el["response"] for el in context_response_data]
                if self.triplet_mode:
                    negative_response = self._create_neg_resp_rand(context_response_data, batch_size)
                    if self.hard_triplets_sampling:
                        labels = [el["label"] for el in context_response_data]
                        positives = [random.choices(el["pos_pool"], k=self.num_positive_samples)
                                     for el in context_response_data]
                        x = [[(context[i], el) for el in positives[i]] for i in range(len(context_response_data))]
                        y = labels
                    else:
                        x = [[(context[i], el) for el in [response[i]] + [negative_response[i]]]
                             for i in range(len(context_response_data))]
                        y = batch_size * [np.ones(2)]
                else:
                    y = [el["label"] for el in context_response_data]
                    x = [[(context[i], response[i])] for i in range(len(context_response_data))]
                yield (x, y)
        if data_type in ["valid", "test"]:
            for i in range(num_steps + 1):
                if i < num_steps:
                    context_response_data = data[i * batch_size:(i + 1) * batch_size]
                else:
                    if len(data[i * batch_size:len(data)]) > 0:
                        context_response_data = data[i * batch_size:len(data)]
                context = [el["context"] for el in context_response_data]
                if data_type == "valid":
                    ranking_length = self.num_ranking_samples_valid
                    sample_candidates_pool = self.sample_candidates_pool_valid
                elif data_type == "test":
                    ranking_length = self.num_ranking_samples_test
                    sample_candidates_pool = self.sample_candidates_pool_test
                if not sample_candidates_pool:
                    ranking_length = self.len_vocab
                response_data = self._create_rank_resp(context_response_data, ranking_length)
                if self.pos_pool_rank:
                    y = [len(el["pos_pool"]) * np.ones(ranking_length) for el in context_response_data]
                else:
                    y = [np.ones(ranking_length) for _ in context_response_data]
                x = [[(context[i], el) for el in response_data[i]] for i in range(len(context_response_data))]
                yield (x, y)

    def _create_neg_resp_rand(self, context_response_data, batch_size):
        """Randomly chooses negative response for each context in a batch.

        Sampling is performed from predefined pools of candidates or from the whole data.

        Args:
            context_response_data: A batch from the train part of the dataset.
            batch_size: A batch size.

        Returns:
            one negative response for each context in a batch.
        """
        if self.sample_candidates_pool:
            negative_response_data = [random.choice(el["neg_pool"])
                                      for el in context_response_data]
        else:
            candidates = []
            for i in range(batch_size):
                candidate = np.random.randint(0, self.len_vocab, 1)[0]
                while candidate in context_response_data[i]["pos_pool"]:
                    candidate = np.random.randint(0, self.len_vocab, 1)[0]
                candidates.append(candidate)
            negative_response_data = candidates
        return negative_response_data

    def _create_rank_resp(self, context_response_data, ranking_length):
        """Chooses a set of negative responses for each context in a batch to evaluate ranking quality.

        Negative responses are taken from predefined pools of candidates or from the whole data.

        Args:
            context_response_data: A batch from the train part of the dataset.
            ranking_length: a number of responses for each context to evaluate ranking quality.

        Returns:
            list of responses for each context in a batch.
        """
        response_data = []
        for i in range(len(context_response_data)):
            pos_pool = context_response_data[i]["pos_pool"]
            resp = context_response_data[i]["response"]
            if self.pos_pool_rank:
                pos_pool.insert(0, pos_pool.pop(pos_pool.index(resp)))
            else:
                pos_pool = [resp]
            neg_pool = context_response_data[i]["neg_pool"]
            response = pos_pool + neg_pool
            response_data.append(response[:ranking_length])
        return response_data
