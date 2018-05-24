from abc import ABCMeta, abstractmethod
import numpy as np
from deeppavlov.core.commands.utils import expand_path
from keras.preprocessing.sequence import pad_sequences
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


class RankingDict(metaclass=ABCMeta):

    def __init__(self, save_path, load_path,
                 max_sequence_length, padding, truncating):

        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.truncating = truncating

        save_path = expand_path(save_path).resolve().parent
        load_path = expand_path(load_path).resolve().parent

        self.tok_save_path = save_path / "tok2int.dict"
        self.tok_load_path = load_path / "tok2int.dict"
        self.cont_save_path = save_path / "cont2toks.dict"
        self.cont_load_path = load_path / "cont2toks.dict"
        self.resp_save_path = save_path / "resp2toks.dict"
        self.resp_load_path = load_path / "resp2toks.dict"
        self.cemb_save_path = str(save_path / "context_embs.npy")
        self.cemb_load_path = str(load_path / "context_embs.npy")
        self.remb_save_path = str(save_path / "response_embs.npy")
        self.remb_load_path = str(load_path / "response_embs.npy")

        self.int2tok_vocab = {}
        self.tok2int_vocab = {}
        self.response2toks_vocab = {}
        self.response2emb_vocab = {}
        self.context2toks_vocab = {}
        self.context2emb_vocab = {}

    def init_from_scratch(self):
        log.info("[initializing new `{}`]".format(self.__class__.__name__))
        self.build_int2tok_vocab()
        self.build_tok2int_vocab()
        self.build_context2toks_vocabulary()
        self.build_context2emb_vocabulary()
        self.build_response2toks_vocabulary()
        self.build_response2emb_vocabulary()

    def load(self):
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        self.load_int2tok()
        self.build_tok2int_vocab()
        self.load_context2toks()
        self.build_context2emb_vocabulary()
        self.load_response2toks()
        self.build_response2emb_vocabulary()

    def save(self):
        log.info("[saving `{}`]".format(self.__class__.__name__))
        self.save_int2tok()
        self.save_context2toks()
        self.save_response2toks()

    @abstractmethod
    def build_int2tok_vocab(self):
        pass

    @abstractmethod
    def build_response2toks_vocabulary(self):
        pass

    @abstractmethod
    def build_context2toks_vocabulary(self):
        pass

    def build_tok2int_vocab(self):
        self.tok2int_vocab = {el[1]: el[0] for el in self.int2tok_vocab.items()}

    def build_response2emb_vocabulary(self):
        for i in range(len(self.response2toks_vocab)):
            self.response2emb_vocab[i] = None

    def build_context2emb_vocabulary(self):
        for i in range(len(self.context2toks_vocab)):
            self.context2emb_vocab[i] = None

    def conts2toks(self, conts_li):
        toks_li = [self.context2toks_vocab[cont] for cont in conts_li]
        return toks_li

    def resps2toks(self, resps_li):
        toks_li = [self.response2toks_vocab[resp] for resp in resps_li]
        return toks_li

    def make_toks(self, items_li, type):
        if type == "context":
            toks_li = self.conts2toks(items_li)
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
            f.write(
                '\n'.join(['\t'.join([str(el[0]), ' '.join(el[1])]) for el in self.response2toks_vocab.items()]))

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

