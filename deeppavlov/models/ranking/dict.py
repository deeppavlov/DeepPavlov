import numpy as np
from pathlib import Path
from deeppavlov.core.commands.utils import expand_path
from keras.preprocessing.sequence import pad_sequences

class InsuranceDict(object):

    def __init__(self, vocabs_path, save_path, load_path,
                 max_sequence_length, padding="post", truncating="pre",
                 ):
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.truncating = truncating

        save_path = expand_path(save_path).resolve().parent
        load_path = expand_path(load_path).resolve().parent

        self.vocabs_path = expand_path(vocabs_path)
        self.tok_save_path = save_path / "tok2int.dict"
        self.tok_load_path = load_path / "tok2int.dict"
        self.cont_save_path = save_path / "cont2toks.dict"
        self.cont_load_path = load_path / "cont2toks.dict"
        self.resp_save_path = save_path / "resp2toks.dict"
        self.resp_load_path = load_path / "resp2toks.dict"
        self.cemb_save_path =str(save_path / "context_embs.npy")
        self.cemb_load_path =str(load_path / "context_embs.npy")
        self.remb_save_path =str(save_path / "response_embs.npy")
        self.remb_load_path =str(load_path / "response_embs.npy")

        self.int2tok_vocab = {}
        self.tok2int_vocab = {}
        self.response2toks_vocab = {}
        self.response2emb_vocab = {}
        self.context2toks_vocab = {}
        self.context2emb_vocab = {}

    def init_from_scratch(self):
        int2tok_fname = Path(self.vocabs_path) / 'vocabulary'
        self.build_int2tok_vocab(int2tok_fname)
        self.build_tok2int_vocab()
        response2ints_fname = Path(self.vocabs_path) / 'answers.label.token_idx'
        self.build_response2toks_vocabulary(response2ints_fname)
        self.build_response2emb_vocabulary()
        context2ints_fname = Path(self.vocabs_path) / 'question.train.token_idx.label'
        self.build_context2toks_vocabulary(context2ints_fname)
        self.build_context2emb_vocabulary()

    def load(self):
        self.load_int2tok()
        self.build_tok2int_vocab()
        self.load_context2toks()
        self.load_cont()
        self.load_response2toks()
        self.load_resp()

    def save(self):
        self.save_int2tok()
        self.save_context2toks()
        self.save_cont()
        self.save_response2toks()
        self.save_resp()

    def build_int2tok_vocab(self, fname):
        with open(fname) as f:
            data = f.readlines()
        self.int2tok_vocab = {int(el.split('\t')[0].split('_')[1]): el.split('\t')[1][:-1] for el in data}
        self.int2tok_vocab[0] = '<UNK>'

    def build_tok2int_vocab(self):
        self.tok2int_vocab = {el[1]: el[0] for el in self.int2tok_vocab.items()}

    def build_response2toks_vocabulary(self, fname):
        with open(fname, 'r') as f:
            data = f.readlines()
            response2idxs_vocab = {int(el.split('\t')[0]) - 1:
                                   (el.split('\t')[1][:-1]).split(' ') for el in data}
        self.response2toks_vocab = {el[0]: [self.int2tok_vocab[int(x.split('_')[1])]
                                    for x in el[1]] for el in response2idxs_vocab.items()}

    def build_context2toks_vocabulary(self, fname):
        contexts = []
        with open(fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            c, _ = eli.split('\t')
            contexts.append(c.split(' '))

        self.context2toks_vocab = {el[0]: [self.int2tok_vocab[int(x.split('_')[1])]
                                   for x in el[1]] for el in enumerate(contexts)}

    def build_response2emb_vocabulary(self):
        for i in range(len(self.response2toks_vocab)):
            self.response2emb_vocab[i] = None

    def build_context2emb_vocabulary(self):
        for i in range(len(self.context2toks_vocab)):
            self.context2emb_vocab[i] = None

    def ints2toks(self, idxs_li):
        toks_li = []
        for el in idxs_li:
            toks = [self.int2tok_vocab[int] for int in el]
            toks_li.append(toks)
        return toks_li

    def resps2toks(self, resps_li):
        toks_li = [self.response2toks_vocab[resp] for resp in resps_li]
        return toks_li

    def make_toks(self, items_li, type):
        if type == "context":
            toks_li = self.ints2toks(items_li)
        elif type == "response":
            toks_li = self.resps2toks(items_li)
        return toks_li

    def make_ints(self, toks_li):
        ints_li = []
        for toks in toks_li:
            ints = []
            for tok in toks:
                index = self.tok2int_vocab.get(tok)
                if self.tok2int_vocab.get(tok) is not None:
                    ints.append(index)
                else:
                    ints.append(0)
            ints_li.append(ints)
        ints_li = pad_sequences(ints_li,
                                maxlen=self.max_sequence_length,
                                padding=self.padding,
                                truncating=self.truncating)
        return ints_li

    def save_int2tok(self):
        with self.tok_save_path.open('w') as f:
            f.write('\n'.join(['\t'.join([str(el[0]), el[1]]) for el in self.int2tok_vocab.items()]))

    def load_int2tok(self):
        with self.tok_load_path.open('r') as f:
            data = f.readlines()
        self.int2tok_vocab = {int(el.split('\t')[0]): el.split('\t')[1][:-1] for el in data}

    def save_context2toks(self):
        with self.cont_save_path.open('w') as f:
            f.write('\n'.join(['\t'.join([str(el[0]), ' '.join(el[1])]) for el in self.context2toks_vocab.items()]))

    def load_context2toks(self):
        with self.cont_load_path.open('r') as f:
            data = f.readlines()
        self.context2toks_vocab = {int(el.split('\t')[0]): el.split('\t')[1][:-1].split(' ') for el in data}

    def save_response2toks(self):
        with self.resp_save_path.open('w') as f:
            f.write('\n'.join(['\t'.join([str(el[0]), ' '.join(el[1])]) for el in self.response2toks_vocab.items()]))

    def load_response2toks(self):
        with self.resp_load_path.open('r') as f:
            data = f.readlines()
        self.response2toks_vocab = {int(el.split('\t')[0]): el.split('\t')[1][:-1].split(' ') for el in data}

    def save_cont(self):
        context_embeddings = []
        for i in range(len(self.context2emb_vocab)):
            context_embeddings.append(self.context2emb_vocab[i])
        context_embeddings = np.vstack(context_embeddings)
        np.save(self.cemb_save_path, context_embeddings)

    def load_cont(self):
        context_embeddings_arr = np.load(self.cemb_load_path)
        for i in range(context_embeddings_arr.shape[0]):
            self.context2emb_vocab[i] = context_embeddings_arr[i]

    def save_resp(self):
        response_embeddings = []
        for i in range(len(self.response2emb_vocab)):
            response_embeddings.append(self.response2emb_vocab[i])
        response_embeddings = np.vstack(response_embeddings)
        np.save(self.remb_save_path, response_embeddings)

    def load_resp(self):
        response_embeddings_arr = np.load(self.remb_load_path)
        for i in range(response_embeddings_arr.shape[0]):
            self.response2emb_vocab[i] = response_embeddings_arr[i]

