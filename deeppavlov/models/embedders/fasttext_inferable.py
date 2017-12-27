import urllib.request
import numpy as np
from pathlib import Path
from overrides import overrides

from gensim.models.wrappers.fasttext import FastText

from deeppavlov.core.models.inferable import Inferable


class EmbeddingInferableModel(Inferable):

    def __init__(self, embedding_fname, embedding_dim, embedding_url=None,  *args, **kwargs):
        """
        Method initialize the class according to given parameters.
        Args:
            embedding_fname: name of file with embeddings
            embedding_dim: dimension of embeddings
            embedding_url: url link to embedding to try to download if file does not exist
            *args:
            **kwargs:
        """
        self.tok2emb = {}
        self.embedding_dim = embedding_dim
        self.model = None
        self.load(embedding_fname, embedding_url)

    def add_items(self, sentence_li):
        """
        Method adds new items to tok2emb dictionary from given text
        Args:
            sentence_li: list of sentences

        Returns: None

        """
        for sen in sentence_li:
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            for tok in tokens:
                if self.tok2emb.get(tok) is None:
                    try:
                        self.tok2emb[tok] = self.model[tok]
                    except KeyError:
                        self.tok2emb[tok] = np.zeros(self.embedding_dim)
        return

    def emb2str(self, vec):
        """
        Method returns string corresponding to the given embedding vectors
        Args:
            vec: vector of embeddings

        Returns:
            string corresponding to the given embeddings
        """
        string = ' '.join([str(el) for el in vec])
        return string

    def load(self, embedding_fname, embedding_url=None, *args, **kwargs):
        """
        Method initializes dict of embeddings from file
        Args:
            fname: file name

        Returns:
            Nothing
        """

        if not embedding_fname:
            raise RuntimeError('No pretrained fasttext model provided')
        fasttext_model_file = embedding_fname

        if not Path(fasttext_model_file).is_file():
            emb_path = embedding_url
            if not emb_path:
                raise RuntimeError('No pretrained fasttext model provided')
            embedding_fname = Path(fasttext_model_file).name
            try:
                print('Trying to download a pretrained fasttext model from repository')
                url = urllib.parse.urljoin(emb_path, embedding_fname)
                urllib.request.urlretrieve(url, fasttext_model_file)
                print('Downloaded a fasttext model')
            except Exception as e:
                raise RuntimeError('Looks like the `EMBEDDINGS_URL` variable is set incorrectly', e)
        self.model = FastText.load_fasttext_format(fasttext_model_file)
        return

    @overrides
    def infer(self, instance, *args, **kwargs):
        """
        Method returns embedded data
        Args:
            instance: sentence or list of sentences

        Returns:
            Embedded sentence or list of embedded sentences
        """
        if type(instance) is str:
            tokens = instance.split(" ")
            self.add_items(tokens)
            embedded_tokens = []
            for tok in tokens:
                embedded_tokens.append(self.tok2emb.get(tok))
            if len(tokens) == 1:
                embedded_tokens = embedded_tokens[0]
            return embedded_tokens
        else:
            embedded_instance = []
            for sample in instance:
                tokens = sample.split(" ")
                self.add_items(tokens)
                embedded_tokens = []
                for tok in tokens:
                    embedded_tokens.append(self.tok2emb.get(tok))
                embedded_instance.append(embedded_tokens)
            return embedded_instance

