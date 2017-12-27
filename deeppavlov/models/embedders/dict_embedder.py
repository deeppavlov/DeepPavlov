import numpy as np
from pathlib import Path
from overrides import overrides

from deeppavlov.core.models.inferable import Inferable


class DictEmbedder(Inferable):
    def __init__(self, embedding_dim, model_path=None, *args, **kwargs):
        """
        Method initializes the class according to given parameters.
        Args:
            embedding_dim: embedding dimension
        """
        self.tok2emb = {}
        self.embedding_dim = embedding_dim
        self.load(fname=model_path)

    def load(self, fname):
        """
        Method initializes dictionary of embeddings from file.
        Returns:
            Nothing
        """

        if fname is None or not Path(fname).is_file():
            raise IOError('There is no dictionary of embeddings <<{}>> file provided.'.format(fname))
        else:
            print('Loading existing dictionary of embeddings from {}'.format(fname))
            with open(fname, 'r') as f:
                for line in f:
                    values = line.rsplit(sep=' ', maxsplit=self.embedding_dim)
                    assert(len(values) == self.embedding_dim + 1)
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.tok2emb[word] = coefs

    @overrides
    def infer(self, instance, *args, **kwargs):
        """
        Method returns embedded sentence
        Args:
            instance: string (e.g. "I want some food")

        Returns:
            embedded sentence
        """
        embedded_sentence = []
        for word in instance.split(" "):
            embedded_sentence.append(self.tok2emb[word])
        return embedded_sentence
