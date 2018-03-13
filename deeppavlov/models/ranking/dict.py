import numpy as np
from pathlib import Path
from deeppavlov.core.commands.utils import expand_path
from keras.preprocessing.sequence import pad_sequences

class InsuranceDict(object):

    def __init__(self, vocabs_path, max_sequence_length, padding="post", truncating="pre"):
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.truncating = truncating

        vocabs_path =expand_path(vocabs_path)
        self.idx2tok_vocab = {}
        self.label2toks_vocab = {}
        self.label2emb_vocab = {}
        idx2tok_fname = Path(vocabs_path) / 'vocabulary'
        self.build_idx2tok_vocab(idx2tok_fname)
        self.build_tok2int_vocab()
        label2idxs_fname = Path(vocabs_path) / 'answers.label.token_idx'
        self.build_label2toks_vocabulary(label2idxs_fname)
        self.build_label2emb_vocabulary()
        self.context2toks_vocab = {}
        self.context2emb_vocab = {}
        context2idxs_fname = Path(vocabs_path) / 'question.train.token_idx.label'
        self.build_context2toks_vocabulary(context2idxs_fname)
        self.build_context2emb_vocabulary()

    def build_idx2tok_vocab(self, fname):
        with open(fname) as f:
            data = f.readlines()
            self.idx2tok_vocab = {el.split('\t')[0]: el.split('\t')[1][:-1] for el in data}

    def build_tok2int_vocab(self):
        """Add new items to the tok2emb dictionary from a given text."""
        toks = ['<UNK>'] + list(self.idx2tok_vocab.values())
        self.tok2int = {el[1]: el[0] for el in enumerate(toks)}

    def build_label2toks_vocabulary(self, fname):
        with open(fname, 'r') as f:
            data = f.readlines()
            label2idxs_vocab = {int(el.split('\t')[0]) - 1:
                                        (el.split('\t')[1][:-1]).split(' ') for el in data}
        for el in label2idxs_vocab.items():
            self.label2toks_vocab[el[0]] = [self.idx2tok_vocab[idx] for idx in el[1]]

    def build_context2toks_vocabulary(self, fname):
        contexts = []
        with open(fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            c, _ = eli.split('\t')
            contexts.append(c.split(' '))

        for el in enumerate(contexts):
            self.context2toks_vocab[el[0]] = [self.idx2tok_vocab[idx] for idx in el[1]]

    def build_label2emb_vocabulary(self):
        for i in range(len(self.label2toks_vocab)):
            self.label2emb_vocab[i] = None

    def build_context2emb_vocabulary(self):
        for i in range(len(self.context2toks_vocab)):
            self.context2emb_vocab[i] = None

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

    def make_ints(self, toks_li):
        ints_li = []
        for toks in toks_li:
            ints = []
            for tok in toks:
                index = self.tok2int.get(tok)
                if self.tok2int.get(tok) is not None:
                    ints.append(index)
                else:
                    ints.append(0)
            ints_li.append(ints)
        ints_li = pad_sequences(ints_li,
                                maxlen=self.max_sequence_length,
                                padding=self.padding,
                                truncating=self.truncating)
        return ints_li


    def save_resp(self, path):
        response_embeddings = []
        for i in range(len(self.label2emb_vocab)):
            response_embeddings.append(self.label2emb_vocab[i])
        response_embeddings = np.vstack(response_embeddings)
        np.save(path, response_embeddings)

    def save_cont(self, path):
        context_embeddings = []
        for i in range(len(self.context2emb_vocab)):
            context_embeddings.append(self.context2emb_vocab[i])
        context_embeddings = np.vstack(context_embeddings)
        np.save(path, context_embeddings)

    def load_resp(self, path):
        response_embeddings_arr = np.load(path)
        for i in range(response_embeddings_arr.shape[0]):
            self.label2emb_vocab[i] = response_embeddings_arr[i]

    def load_cont(self, path):
        context_embeddings_arr = np.load(path)
        for i in range(context_embeddings_arr.shape[0]):
            self.context2emb_vocab[i] = context_embeddings_arr[i]
