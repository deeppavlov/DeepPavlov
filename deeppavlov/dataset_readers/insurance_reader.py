from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done
from deeppavlov.core.commands.utils import get_deeppavlov_root, expand_path
from typing import Dict, List, Union


@register('insurance_reader')
class InsuranceReader(DatasetReader):

    def read(self, data_path: str, num_ranking_samples: int = 10, **kwargs) -> Dict[
        str, List[Dict[str, Union[int, List[int]]]]]:
        """Read the InsuranceQA data from files and forms the dataset.

        Args:
            data_path: A path to a folder where dataset files are stored.
            **kwargs: Other parameters.

        Returns:
        data: A dictionary containing training, validation and test parts of the dataset obtainable via
            ``train``, ``valid`` and ``test`` keys.
        """
        self.num_ranking_samples = num_ranking_samples
        data_path = expand_path(data_path)
        self._download_data(data_path)
        dataset = {'train': None, 'valid': None, 'test': None}
        train_fname = Path(data_path) / 'insuranceQA-master/V1/question.train.token_idx.label'
        valid_fname = Path(data_path) / 'insuranceQA-master/V1/question.dev.label.token_idx.pool'
        test_fname = Path(data_path) / 'insuranceQA-master/V1/question.test1.label.token_idx.pool'
        int2tok_fname = Path(data_path) / 'insuranceQA-master/V1/vocabulary'
        response2ints_fname = Path(data_path) / 'insuranceQA-master/V1/answers.label.token_idx'
        self.int2tok_vocab = self._build_int2tok_vocab(int2tok_fname)
        self.idxs2cont_vocab = self._build_context2toks_vocab(train_fname, valid_fname, test_fname)
        self.response2str_vocab = self._build_response2str_vocab(response2ints_fname)
        dataset["valid"] = self._preprocess_data_valid_test(valid_fname)
        dataset["train"] = self._preprocess_data_train(train_fname)
        dataset["test"] = self._preprocess_data_valid_test(test_fname)

        return dataset

    def _download_data(self, data_path):
        """Download archive with the InsuranceQA dataset files and decompress if there is no dataset files in `data_path`.

        Args:
            data_path: A path to a folder where dataset files are stored.
        """
        if not is_done(Path(data_path)):
            download_decompress(url="http://lnsigo.mipt.ru/export/datasets/insuranceQA-master.zip",
                                download_path=data_path)
            mark_done(data_path)

    def _build_context2toks_vocab(self, train_f, val_f, test_f):
        contexts = []
        with open(train_f, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            c, _ = eli.split('\t')
            contexts.append(c)
        with open(val_f, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            _, c, _ = eli.split('\t')
            contexts.append(c)
        with open(test_f, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            _, c, _ = eli.split('\t')
            contexts.append(c)
        idxs2cont_vocab = {el[1]: el[0] for el in enumerate(contexts)}
        return idxs2cont_vocab

    def _build_int2tok_vocab(self, fname):
        with open(fname, 'r') as f:
            data = f.readlines()
        int2tok_vocab = {int(el.split('\t')[0].split('_')[1]): el.split('\t')[1][:-1] for el in data}
        return int2tok_vocab

    def _build_response2str_vocab(self, fname):
        with open(fname, 'r') as f:
            data = f.readlines()
            response2idxs_vocab = {int(el.split('\t')[0]) - 1:
                                       (el.split('\t')[1][:-1]).split(' ') for el in data}
        response2str_vocab = {el[0]: ' '.join([self.int2tok_vocab[int(x.split('_')[1])]
                                               for x in el[1]]) for el in response2idxs_vocab.items()}
        return response2str_vocab

    def _preprocess_data_train(self, fname):
        positive_responses_pool = []
        contexts = []
        responses = []
        labels = []
        with open(fname, 'r') as f:
            data = f.readlines()
        for k, eli in enumerate(data):
            eli = eli[:-1]
            q, pa = eli.split('\t')
            q_tok = ' '.join([self.int2tok_vocab[int(el.split('_')[1])] for el in q.split()])
            pa_list = [int(el) - 1 for el in pa.split(' ')]
            pa_list_tok = [self.response2str_vocab[el] for el in pa_list]
            for elj in pa_list_tok:
                contexts.append(q_tok)
                responses.append(elj)
                positive_responses_pool.append(pa_list_tok)
                labels.append(k)
        train_data = list(zip(contexts, responses))
        train_data = list(zip(train_data, labels))
        return train_data

    def _preprocess_data_valid_test(self, fname):
        pos_responses_pool = []
        neg_responses_pool = []
        contexts = []
        pos_responses = []
        with open(fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            pa, q, na = eli.split('\t')
            q_tok = ' '.join([self.int2tok_vocab[int(el.split('_')[1])] for el in q.split()])
            pa_list = [int(el) - 1 for el in pa.split(' ')]
            pa_list_tok = [self.response2str_vocab[el] for el in pa_list]
            nas = [int(el) - 1 for el in na.split(' ')]
            nas_tok = [self.response2str_vocab[el] for el in nas]
            for elj in pa_list_tok:
                contexts.append(q_tok)
                pos_responses.append(elj)
                pos_responses_pool.append(pa_list_tok)
                neg_responses_pool.append(nas_tok)
        data = [[el[0]] + el[1] for el in zip(contexts, neg_responses_pool)]
        data = [(el[0], len(el[1])) for el in zip(data, pos_responses_pool)]
        return data
