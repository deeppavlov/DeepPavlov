import numpy as np
from gensim.models.wrappers import FastText
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.utils import download


class EmbeddingsDict(object):
    """The class provides embeddings using fasttext model.

    Attributes:
        tok2emb: a dictionary containing embedding vectors (value) for tokens (keys)
        embedding_dim: a dimension of embeddings
        opt: given parameters
        fasttext_model_file: a file containing fasttext binary model
    """

    def __init__(self, toks, emb_model_file, embedding_dim, max_sequence_length, download_url=None,
                 emb_vocab_file=None, embeddings="word2vec", padding="post", truncating="pre"):
        """Initialize the class according to given parameters."""
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.truncating = truncating
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.emb_model_file = expand_path(emb_model_file)
        self.emb_vocab_file = expand_path(emb_vocab_file)
        if not self.emb_model_file.is_file():
            download(source_url=download_url, dest_file_path=self.emb_model_file)

        self.tok2emb = {None: list(np.zeros(self.embedding_dim))}
        self.tok2int = {None: 0}
        self.emb_index = 1

        self.load_items()

        if self.embeddings == "fasttext":
            self.embeddings_model = FastText.load_fasttext_format(str(self.emb_model_file))
        elif self.embeddings == "word2vec":
            self.embeddings_model = Word2Vec.load(str(self.emb_model_file))

        self.add_items(toks)
        self.create_emb_matrix()
        self.save_items()

    def add_items(self, toks_li):
        """Add new items to the tok2emb dictionary from a given text."""
        dummy_emb = list(np.zeros(self.embedding_dim))
        for tok in toks_li:
            if self.tok2emb.get(tok) is None:
                try:
                    self.tok2emb[tok] = self.embeddings_model[tok]
                except:
                    self.tok2emb[tok] = dummy_emb
            if self.tok2int.get(tok) is None:
                self.tok2int[tok] = self.emb_index
                self.emb_index += 1

    def save_items(self):
        """Save the dictionary tok2emb to the file."""
        if not self.emb_vocab_file.is_file():
            with self.emb_vocab_file.open('w') as f:
                data = '\n'.join([el[0] + ' ' + self.emb2str(el[1]) for el in self.tok2emb.items()])
                f.write(data)

    def emb2str(self, vec):
        """Return the string corresponding to given embedding vectors"""
        data = ' '.join([str(el) for el in vec])
        return data

    def load_items(self):
        """Initialize embeddings from the file."""
        if self.emb_vocab_file.is_file():
            with self.emb_vocab_file.open('r') as f:
                for line in f:
                    values = line.rsplit(sep=' ', maxsplit=self.embedding_dim)
                    assert(len(values) == self.embedding_dim + 1)
                    tok = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.tok2emb[tok] = coefs

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

    def create_emb_matrix(self):
        self.emb_matrix = np.zeros((self.emb_index, self.embedding_dim))
        for tok, i in self.tok2int.items():
            self.emb_matrix[i] = self.tok2emb.get(tok)
