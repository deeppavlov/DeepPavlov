import numpy as np
from gensim.models.wrappers import FastText
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
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

    def __init__(self, tok2int_vocab, embedding_dim, embeddings="word2vec"):
        """Initialize the class according to given parameters."""
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.emb_model_file = next(expand_path("insurance_embeddings").iterdir())

        if self.embeddings == "fasttext":
            self.embeddings_model = FastText.load_fasttext_format(str(self.emb_model_file))
        elif self.embeddings == "word2vec":
            self.embeddings_model = Word2Vec.load(str(self.emb_model_file))

        self.create_emb_matrix(tok2int_vocab)

    def create_emb_matrix(self, tok2int_vocab):
        dummy_emb = list(np.zeros(self.embedding_dim))
        self.emb_matrix = np.zeros((len(tok2int_vocab), self.embedding_dim))
        for tok, i in tok2int_vocab.items():
            if tok == '<UNK>':
                self.emb_matrix[i] = dummy_emb
            else:
                try:
                    self.emb_matrix[i] = self.embeddings_model[tok]
                except:
                    self.emb_matrix[i] = dummy_emb
