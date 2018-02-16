from pathlib import Path
from deeppavlov.core.commands.utils import expand_path


class InsuranceDict(object):

    def __init__(self, vocabs_path):
        vocabs_path =expand_path(vocabs_path)
        self.idx2tok_vocab = {}
        self.label2toks_vocab = {}
        idx2tok_fname = Path(vocabs_path) / 'vocabulary'
        self.build_idx2tok_vocab(idx2tok_fname)
        label2idxs_fname = Path(vocabs_path) / 'answers.label.token_idx'
        self.build_label2toks_vocabulary(label2idxs_fname)
        self.toks = [el[1] for el in self.idx2tok_vocab.items()]

    def build_idx2tok_vocab(self, fname):
        with open(fname) as f:
            data = f.readlines()
            self.idx2tok_vocab = {el.split('\t')[0]: el.split('\t')[1][:-1] for el in data}

    def build_label2toks_vocabulary(self, fname):
        with open(fname, 'r') as f:
            data = f.readlines()
            label2idxs_vocab = {int(el.split('\t')[0]) - 1:
                                        (el.split('\t')[1][:-1]).split(' ') for el in data}
        for el in label2idxs_vocab.items():
            self.label2toks_vocab[el[0]] = [self.idx2tok_vocab[idx] for idx in el[1]]

    def idxs2toks(self, idxs_li):
        toks_li = []
        for el in idxs_li:
            toks = [self.idx2tok_vocab[idx] for idx in el]
            toks_li.append(toks)
        return toks_li

    def labels2toks(self, labels_li):
        toks_li = [self.label2toks_vocab[label] for label in labels_li]
        return toks_li

    def make_toks(self, items_li, type):
        if type == "context":
            toks_li = self.idxs2toks(items_li)
        elif type == "response":
            toks_li = self.labels2toks(items_li)
        return toks_li
