import os
import numpy as np
from gensim.models.wrappers import FastText
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from deeppavlov.core.commands.utils import expand_path


class EmbeddingsDict(object):
    """The class provides embeddings using fasttext model.

    Attributes:
        tok2emb: a dictionary containing embedding vectors (value) for tokens (keys)
        embedding_dim: a dimension of embeddings
        opt: given parameters
        fasttext_model_file: a file containing fasttext binary model
    """

    def __init__(self, toks, emb_model_file, embedding_dim, max_sequence_length,
                 emb_vocab_file=None, embeddings="word2vec", padding="post", truncating="pre"):
        """Initialize the class according to given parameters."""
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.truncating = truncating
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.emb_model_file = str(expand_path(emb_model_file))
        self.emb_vocab_file = str(expand_path(emb_vocab_file))

        self.tok2emb = {}
        self.tok_index = {}
        self.emb_index = 0

        self.load_items()

        if self.embeddings == "fasttext":
            self.embeddings_model = FastText.load_fasttext_format(self.emb_model_file)
        elif self.embeddings == "word2vec":
            self.embeddings_model = Word2Vec.load(self.emb_model_file)

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
            if self.tok_index.get(tok) is None:
                self.tok_index[tok] = self.emb_index
                self.emb_index += 1

    def save_items(self):
        """Save the dictionary tok2emb to the file."""
        if self.emb_vocab_file is not None and not os.path.isfile(self.emb_vocab_file):
            f = open(self.emb_vocab_file, 'w')
            string = '\n'.join([el[0] + ' ' + self.emb2str(el[1]) for el in self.tok2emb.items()])
            f.write(string)
            f.close()

    def emb2str(self, vec):
        """Return the string corresponding to given embedding vectors"""

        string = ' '.join([str(el) for el in vec])
        return string

    def load_items(self):
        """Initialize embeddings from the file."""

        if self.emb_vocab_file is not None:
            if not os.path.isfile(self.emb_vocab_file):
                print('There is no %s file provided. Initializing new dictionary.' % self.emb_vocab_file)
            else:
                print('Loading existing dictionary  from %s.' % self.emb_vocab_file)
                with open(self.emb_vocab_file, 'r') as f:
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
                ints.append(self.tok_index[tok])
            ints_li.append(ints)
        ints_li = pad_sequences(ints_li,
                                maxlen=self.max_sequence_length,
                                padding=self.padding,
                                truncating=self.truncating)
        return ints_li

    def create_emb_matrix(self):
        self.emb_matrix = np.zeros((self.emb_index, self.embedding_dim))
        for tok, i in self.tok_index.items():
            self.emb_matrix[i] = self.tok2emb.get(tok)


