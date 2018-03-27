import numpy as np
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.utils import download


class Embeddings(object):
    """The class provides embeddings using fasttext model.

    Attributes:
        tok2emb: a dictionary containing embedding vectors (value) for tokens (keys)
        embedding_dim: a dimension of embeddings
        opt: given parameters
        fasttext_model_file: a file containing fasttext binary model
    """

    def __init__(self, tok2int_vocab, embedding_dim, download_url=None, embeddings="word2vec", seed=None):
        """Initialize the class according to given parameters."""
        np.random.seed(seed)
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.emb_model_file = expand_path("pre-trained_embeddings")
        if not self.emb_model_file.is_file():
            download(source_url=download_url, dest_file_path=self.emb_model_file)

        if self.embeddings == "fasttext":
            self.embeddings_model = FastText.load_fasttext_format(str(self.emb_model_file))
        elif self.embeddings == "word2vec":
            self.embeddings_model = KeyedVectors.load_word2vec_format(str(self.emb_model_file),
                                                                      binary=True)

        self.create_emb_matrix(tok2int_vocab)

    def create_emb_matrix(self, tok2int_vocab):
        self.emb_matrix = np.zeros((len(tok2int_vocab), self.embedding_dim))
        for tok, i in tok2int_vocab.items():
            if tok == '<UNK>':
                self.emb_matrix[i] = np.random.uniform(-0.6, 0.6, self.embedding_dim)
            else:
                try:
                    self.emb_matrix[i] = self.embeddings_model[tok]
                except:
                    self.emb_matrix[i] = np.random.uniform(-0.6, 0.6, self.embedding_dim)
