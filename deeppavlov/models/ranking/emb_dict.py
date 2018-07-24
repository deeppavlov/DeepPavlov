import numpy as np
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from deeppavlov.core.commands.utils import expand_path
from pathlib import Path
from deeppavlov.core.data.utils import download
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


class Embeddings(object):
    """The class provides embeddings using fasttext model.

    Attributes:
        tok2emb: a dictionary containing embedding vectors (value) for tokens (keys)
        embedding_dim: a dimension of embeddings
        opt: given parameters
        fasttext_model_file: a file containing fasttext binary model
    """

    def __init__(self, embedding_dim, max_sequence_length,
                 embeddings_path, save_path, load_path, embeddings="word2vec", seed=None):
        """Initialize the class according to given parameters."""
        np.random.seed(seed)
        save_path = expand_path(save_path).resolve().parent
        load_path = expand_path(load_path).resolve().parent
        self.int2emb_save_path = save_path / "int2emb.dict"
        self.int2emb_load_path = load_path / "int2emb.dict"
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.emb_model_file = expand_path(embeddings_path)

        self.int2emb_vocab = {}

    def init_from_scratch(self, tok2int_vocab):
        if self.embeddings == "fasttext":
            self.embeddings_model = FastText.load_fasttext_format(str(self.emb_model_file))
        elif self.embeddings == "word2vec":
            self.embeddings_model = KeyedVectors.load_word2vec_format(str(self.emb_model_file),
                                                                      binary=True)
        log.info("[initializing new `{}`]".format(self.__class__.__name__))
        self.build_int2emb_vocab(tok2int_vocab)
        self.build_emb_matrix(tok2int_vocab)

    def load(self):
        """Initialize embeddings from the file."""
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        if self.int2emb_load_path.is_file():
            with open(self.int2emb_load_path, 'r', encoding='utf8') as f:
                for line in f:
                    values = line.rsplit(sep=' ', maxsplit=self.embedding_dim)
                    assert(len(values) == self.embedding_dim + 1)
                    index = int(values[0])
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.int2emb_vocab[index] = coefs

    def save(self):
        log.info("[saving `{}`]".format(self.__class__.__name__))
        """Save the dictionary tok2emb to the file."""
        if not self.int2emb_save_path.is_file():
            with open(self.int2emb_save_path, 'w', encoding='utf8') as f:
                data = '\n'.join([str(el[0]) + ' ' +
                                 ' '.join(list(map(str, el[1]))) for el in self.int2emb_vocab.items()])
                f.write(data)

    def build_emb_matrix(self, tok2int_vocab):
        self.emb_matrix = np.zeros((len(tok2int_vocab), self.embedding_dim))
        for tok, i in tok2int_vocab.items():
            if tok == '<UNK>':
                self.emb_matrix[i] = np.random.uniform(-0.6, 0.6, self.embedding_dim)
            else:
                try:
                    self.emb_matrix[i] = self.embeddings_model[tok]
                except:
                    self.emb_matrix[i] = np.random.uniform(-0.6, 0.6, self.embedding_dim)

    def build_int2emb_vocab(self, tok2int_vocab):
        for tok, i in tok2int_vocab.items():
            if tok == '<UNK>':
                self.int2emb_vocab[i] = np.random.uniform(-0.6, 0.6, self.embedding_dim)
            else:
                try:
                    self.int2emb_vocab[i] = self.embeddings_model[tok]
                except:
                    self.int2emb_vocab[i] = np.random.uniform(-0.6, 0.6, self.embedding_dim)

    def get_embs(self, ints):
        embs = []
        for el in ints:
            emb = []
            for int_tok in el:
                assert type(int_tok) != int
                emb.append(self.int2emb_vocab[int_tok])
            emb = np.vstack(emb)
            embs.append(emb)
        embs = [np.reshape(el, (1, self.max_sequence_length, self.embedding_dim)) for el in embs]
        embs = np.vstack(embs)
        return embs

