from deeppavlov.core.data.dataset_reader import DatasetReader
from pathlib import Path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done
from deeppavlov.core.commands.utils import get_deeppavlov_root, expand_path
import pickle
import numpy as np
@register('ubuntu_reader')
class UbuntuReader(DatasetReader):

    def read(self, data_path):
        data_path = expand_path(data_path)
        self.download_data(data_path)
        # dataset = {'train': None, 'valid': None, 'test': None}
        fname = Path(data_path) / 'dataset_1MM/dataset.pkl'
        dataset = self.preprocess_data(fname)
        return dataset

    def download_data(self, data_path):
        if not is_done(Path(data_path)):
            download_decompress(url="http://lnsigo.mipt.ru/export/datasets/ubuntu_dialogs.tgz",
                                download_path=data_path)
            mark_done(data_path)

    def preprocess_data(self, fname):

        positive_responses_pool = []
        contexts = []
        responses = []
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        all_resps = data[0]['r'] + data[1]['r'] + data[2]['r']
        all_resps = set([' '.join(map(str, el)) for el in all_resps])
        vocab = {el[1]:el[0] for el in enumerate(all_resps)}

        train_resps = [vocab[' '.join(map(str, el))] for el in data[0]['r']]
        train_data = [[el[0], el[1]] for el in zip(data[0]['c'], train_resps, data[0]['y']) if el[2] == '1']
        train_data = [{"context": el[0], "response": el[1],
                       "pos_pool": el[1], "neg_pool": None}
                      for el in train_data]
        val_resps = [vocab[' '.join(map(str, el))] for el in data[1]['r']]
        pos_resps = []
        neg_resps = []
        neg_resp = []
        for el in zip(val_resps, data[1]['y']):
            if el[1] == '1':
                pos_resps.append(el[0])
                if len(neg_resp) > 0:
                    neg_resps.append(neg_resp)
                    neg_resp = []
            else:
                neg_resp.append(el[0])
        neg_resps.append(neg_resp)


        val_data = [[el[0], el[1]] for el in zip(data[1]['c'], val_resps, data[1]['y']) if el[2] == '1']
        val_data = [{"context": el[0], "response": el[1],
                       "pos_pool": el[1], "neg_pool": None}
                      for el in val_data]



        return train_data
