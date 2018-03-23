import numpy as np
import pickle
from gensim.models.wrappers import FastText
from gensim.models import Word2Vec
from deeppavlov.core.commands.utils import expand_path
from pathlib import Path
from deeppavlov.core.data.utils import download


class Embeddings(object):
    """The class provides embeddings using fasttext model.

    Attributes:
        tok2emb: a dictionary containing embedding vectors (value) for tokens (keys)
        embedding_dim: a dimension of embeddings
        opt: given parameters
        fasttext_model_file: a file containing fasttext binary model
    """

    def __init__(self, tok2int_vocab, embedding_dim, download_url,
                 embeddings_path, embeddings="word2vec"):
        """Initialize the class according to given parameters."""
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.emb_model_file = expand_path(embeddings_path) / Path(download_url).parts[-1]
        if not self.emb_model_file.is_file():
            download(source_url=download_url, dest_file_path=self.emb_model_file)

        if self.embeddings == "fasttext":
            self.embeddings_model = FastText.load_fasttext_format(str(self.emb_model_file))
        elif self.embeddings == "word2vec":
            self.embeddings_model = Word2Vec.load(str(self.emb_model_file))
        elif self.embeddings == "ubuntu":
            with open(self.emb_model_file, 'rb') as f:
                W_data = pickle.load(f, encoding='bytes')
            bwords = [el[0] for el in W_data[1].items()]
            words = ['<UNK>'] + [el.decode("utf-8") for el in bwords]
            self.embeddings_model = {el[0]: el[1] for el in zip(words, W_data[0])}

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
