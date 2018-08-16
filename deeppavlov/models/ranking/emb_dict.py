import numpy as np
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from deeppavlov.core.commands.utils import expand_path
from pathlib import Path
from deeppavlov.core.data.utils import download
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


class EmbDict(object):
    """The class that provides token (word) embeddings.

    Args:
        save_path: A path including filename to store the instance of
            :class:`deeppavlov.models.ranking.ranking_network.RankingNetwork`.
        load_path: A path including filename to load the instance of
            :class:`deeppavlov.models.ranking.ranking_network.RankingNetwork`.
        max_sequence_length: A maximum length of a sequence in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        seed: Random seed.
        embeddings: A type of embeddings. Possible values are ``fasttext``, ``word2vec`` and ``random``.
        embeddings_path: A path to an embeddings model including filename.
            The type of the model should coincide with the type of embeddings defined by the ``embeddings`` parameter.
        embedding_dim: Dimensionality of token (word) embeddings.
        use_matrix: Whether to use trainable matrix with token (word) embeddings.
    """

    def __init__(self,
                 save_path: str,
                 load_path: str,
                 embeddings_path: str,
                 max_sequence_length: int,
                 embedding_dim: int = 300,
                 embeddings: str = "word2vec",
                 seed: int = None,
                 use_matrix: bool = False):

        np.random.seed(seed)
        save_path = expand_path(save_path).resolve().parent
        load_path = expand_path(load_path).resolve().parent
        self.int2emb_save_path = save_path / "int2emb.npy"
        self.int2emb_load_path = load_path / "int2emb.npy"
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.emb_model_file = expand_path(embeddings_path)
        self.use_matrix = use_matrix
        self.emb_matrix = None

    def init_from_scratch(self, tok2int_vocab):
        if self.embeddings == "fasttext":
            self.embeddings_model = FastText.load_fasttext_format(str(self.emb_model_file))
        elif self.embeddings == "word2vec":
            self.embeddings_model = KeyedVectors.load_word2vec_format(str(self.emb_model_file),
                                                                      binary=True)
        elif self.embeddings == "random":
            self.embeddings_model = {el: np.random.uniform(-0.6, 0.6, self.embedding_dim)
                                     for el in tok2int_vocab.keys()}
        log.info("[initializing new `{}`]".format(self.__class__.__name__))
        self.build_emb_matrix(tok2int_vocab)

    def load(self):
        """Initialize embeddings from the file."""
        if not self.use_matrix:
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            if self.int2emb_load_path.is_file():
                self.emb_matrix = np.load(self.int2emb_load_path)

    def save(self):
        """Save the dictionary tok2emb to the file."""
        if not self.use_matrix:
            log.info("[saving `{}`]".format(self.__class__.__name__))
            if not self.int2emb_save_path.is_file():
                np.save(self.int2emb_save_path, self.emb_matrix)

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
        del self.embeddings_model

    def get_embs(self, ints):
        embs = []
        for el in ints:
            emb = []
            for int_tok in el:
                assert type(int_tok) != int
                emb.append(self.emb_matrix[int_tok])
            emb = np.vstack(emb)
            embs.append(emb)
        embs = [np.reshape(el, (1, self.max_sequence_length, self.embedding_dim)) for el in embs]
        embs = np.vstack(embs)
        return embs

