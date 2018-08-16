from abc import ABCMeta, abstractmethod
import numpy as np
from deeppavlov.core.commands.utils import expand_path
from keras.preprocessing.sequence import pad_sequences
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


class RankingDict(metaclass=ABCMeta):
    """Class to encode characters, tokens, whole contexts and responses with vocabularies, to pad and truncate.

    Args:
        save_path: A path including filename to store the instance of
            :class:`deeppavlov.models.ranking.ranking_network.RankingNetwork`.
        load_path: A path including filename to load the instance of
            :class:`deeppavlov.models.ranking.ranking_network.RankingNetwork`.
        max_sequence_length: A maximum length of a sequence in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        tok_dynamic_batch:  Whether to use dynamic batching. If ``True``, a maximum length of a sequence for a batch
            will be equal to the maximum of all sequences lengths from this batch,
            but not higher than ``max_sequence_length``.
        padding: Padding. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a sequence will be padded at the beginning.
            If set to ``post`` it will padded at the end.
        truncating: Truncating. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a sequence will be truncated at the beginning.
            If set to ``post`` it will truncated at the end.
        max_token_length: A maximum length of a token for representing it by a character-level embedding.
        char_dynamic_batch: Whether to use dynamic batching for character-level embeddings.
            If ``True``, a maximum length of a token for a batch
            will be equal to the maximum of all tokens lengths from this batch,
            but not higher than ``max_token_length``.
        char_pad: Character-level padding. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a token will be padded at the beginning.
            If set to ``post`` it will padded at the end.
        char_trunc: Character-level truncating. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a token will be truncated at the beginning.
            If set to ``post`` it will truncated at the end.
        update_embeddings: Whether to store and update context and response embeddings or not.
    """

    def __init__(self,
                 save_path: str,
                 load_path: str,
                 max_sequence_length: int,
                 max_token_length: int,
                 padding: str = 'post',
                 truncating: str = 'post',
                 token_embeddings: bool = True,
                 char_embeddings: bool = False,
                 char_pad: str = 'post',
                 char_trunc: str = 'post',
                 tok_dynamic_batch: bool = False,
                 char_dynamic_batch: bool = False,
                 update_embeddings: bool = False):

        self.max_sequence_length = max_sequence_length
        self.token_embeddings = token_embeddings
        self.char_embeddings = char_embeddings
        self.max_token_length = max_token_length
        self.padding = padding
        self.truncating = truncating
        self.char_pad = char_pad
        self.char_trunc = char_trunc
        self.tok_dynamic_batch = tok_dynamic_batch
        self.char_dynamic_batch = char_dynamic_batch
        self.upd_embs = update_embeddings

        save_path = expand_path(save_path).resolve().parent
        load_path = expand_path(load_path).resolve().parent

        self.char_save_path = save_path / "char2int.dict"
        self.char_load_path = load_path / "char2int.dict"
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
        if self.char_embeddings:
            self.build_int2char_vocab()
            self.build_char2int_vocab()
        self.build_int2tok_vocab()
        self.build_tok2int_vocab()
        self.build_context2toks_vocabulary()
        self.build_response2toks_vocabulary()
        if self.upd_embs:
            self.build_context2emb_vocabulary()
            self.build_response2emb_vocabulary()

    def load(self):
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        if self.char_embeddings:
            self.load_int2char()
            self.build_char2int_vocab()
        self.load_int2tok()
        self.build_tok2int_vocab()
        self.load_context2toks()
        self.load_response2toks()
        if self.upd_embs:
            self.load_cont()
            self.load_resp()

    def save(self):
        log.info("[saving `{}`]".format(self.__class__.__name__))
        if self.char_embeddings:
            self.save_int2char()
        self.save_int2tok()
        self.save_context2toks()
        self.save_response2toks()
        if self.upd_embs:
            self.save_cont()
            self.save_resp()

    @abstractmethod
    def build_int2char_vocab(self):
        pass

    @abstractmethod
    def build_int2tok_vocab(self):
        pass

    @abstractmethod
    def build_response2toks_vocabulary(self):
        pass

    @abstractmethod
    def build_context2toks_vocabulary(self):
        pass

    def build_char2int_vocab(self):
        self.char2int_vocab = {el[1]: el[0] for el in self.int2char_vocab.items()}

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
        if self.tok_dynamic_batch:
            msl = min(max([len(el) for el in toks_li]), self.max_sequence_length)
        else:
            msl = self.max_sequence_length
        if self.char_dynamic_batch:
            mtl = min(max(len(x) for el in toks_li for x in el), self.max_token_length)
        else:
            mtl = self.max_token_length

        if self.token_embeddings and not self.char_embeddings:
            return self.make_tok_ints(toks_li, msl)
        elif not self.token_embeddings and self.char_embeddings:
            return self.make_char_ints(toks_li, msl, mtl)
        elif self.token_embeddings and self.char_embeddings:
            tok_ints = self.make_tok_ints(toks_li, msl)
            char_ints = self.make_char_ints(toks_li, msl, mtl)
            return np.concatenate([np.expand_dims(tok_ints, axis=2), char_ints], axis=2)

    def make_tok_ints(self, toks_li, msl):
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
                                maxlen=msl,
                                padding=self.padding,
                                truncating=self.truncating)
        return ints_li

    def make_char_ints(self, toks_li, msl, mtl):
        ints_li = np.zeros((len(toks_li), msl, mtl))

        for i, toks in enumerate(toks_li):
            if self.truncating == 'post':
                toks = toks[:msl]
            else:
                toks = toks[-msl:]
            for j, tok in enumerate(toks):
                if self.padding == 'post':
                    k = j
                else:
                    k = j + msl - len(toks)
                ints = []
                for char in tok:
                    index = self.char2int_vocab.get(char)
                    if index is not None:
                        ints.append(index)
                    else:
                        ints.append(0)
                if self.char_trunc == 'post':
                    ints = ints[:mtl]
                else:
                    ints = ints[-mtl:]
                if self.char_pad == 'post':
                    ints_li[i, k, :len(ints)] = ints
                else:
                    ints_li[i, k, -len(ints):] = ints
        return ints_li

    def save_int2char(self):
        with self.char_save_path.open('w') as f:
            f.write('\n'.join(['\t'.join([str(el[0]), el[1]]) for el in self.int2char_vocab.items()]))

    def load_int2char(self):
        with self.char_load_path.open('r') as f:
            data = f.readlines()
        self.int2char_vocab = {int(el.split('\t')[0]): el.split('\t')[1][:-1] for el in data}

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

