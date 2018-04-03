import numpy as np
import pickle
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
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

    def __init__(self, tok2int_vocab, embedding_dim, max_sequence_length, download_url,
                 embeddings_path, embeddings="word2vec", seed=None):
        """Initialize the class according to given parameters."""
        np.random.seed(seed)
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.emb_model_file = expand_path(embeddings_path) / Path(download_url).parts[-1]
        if not self.emb_model_file.is_file():
            download(source_url=download_url, dest_file_path=self.emb_model_file)

        if self.embeddings == "fasttext":
            self.embeddings_model = FastText.load_fasttext_format(str(self.emb_model_file))
        elif self.embeddings == "word2vec":
            self.embeddings_model = KeyedVectors.load_word2vec_format(str(self.emb_model_file),
                                                                      binary=True)
        elif self.embeddings == "ubuntu":
            with open(self.emb_model_file, 'rb') as f:
                W_data = pickle.load(f, encoding='bytes')
            bwords = [el[0] for el in W_data[1].items()]
            words = ['<UNK>'] + [el.decode("utf-8") for el in bwords]
            self.embeddings_model = {el[0]: el[1] for el in zip(words, W_data[0])}

        self.int2emb_vocab = {}
        self.build_seq_embs(tok2int_vocab)
        self.build_emb_matrix(tok2int_vocab)

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

    def build_seq_embs(self, tok2int_vocab):
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
        # assert len(embs) > 1
        embs = np.vstack(embs)
        return embs
