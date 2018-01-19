import urllib
from pathlib import Path

import numpy as np
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.inferable import Inferable


@register('fasttext')
class FasttextEmbedder(Inferable):
    def __init__(self, ser_path=None, ser_dir=None, ser_file=None, dim=100,
                 embedding_url=None, emb_module='fasttext', *args, **kwargs):
        """
        Args:
            ser_path: path to binary file with embeddings
            dim: dimension of embeddings
            embedding_url: url link to embedding to try to download if file does not exist
        """
        super().__init__(ser_path=ser_path,
                         ser_dir=ser_dir,
                         ser_file=ser_file)
        self.tok2emb = {}
        self.dim = dim
        self.embedding_url = embedding_url
        self.emb_module = emb_module
        self.model = self.load()

    def emb2str(self, vec):
        """
        Return string corresponding to the given embedding vectors
        Args:
            vec: vector of embeddings

        Returns:
            string corresponding to the given embeddings
        """
        return ' '.join([str(el) for el in vec])

    def load(self, *args, **kwargs):
        """
        Load dict of embeddings from file
        Args:
            fname: file name
        """

        print("[loading embeddings from `{}`]".format(self.ser_path))
        if not Path(self.ser_path).exists():
            if self.embedding_url:
                try:
                    print('[trying to download a pretrained fasttext model from repository]')
                    local_filename, _ = urllib.request.urlretrieve(self.embedding_url)
                    with open(local_filename, 'rb') as fin:
                        model_file = fin.read()

                    mp = self.ser_path / self._ser_dir / self._ser_file
                    print("[saving downloaded fasttext model to {}]".format(mp))
                    if not mp.exists():
                        mp.mkdir()
                    with open(str(mp), 'wb') as fout:
                        fout.write(model_file)
                except Exception as e:
                    raise RuntimeError(
                        'Looks like the provided fasttext url is incorrect', e)
            else:
                raise FileNotFoundError(
                    'No pretrained fasttext model provided or provided "ser_path" is incorrect.'
                    ' Please include "ser_path" to json.')
        else:
            model_file = str(self.ser_path)
        if self.emb_module == 'fasttext':
            import fasttext as Fasttext
            model = Fasttext.load_model(model_file)
        elif self.emb_module == 'pyfasttext':
            from pyfasttext import FastText as Fasttext
            model = Fasttext(model_file)
        else:
            from gensim.models.wrappers.fasttext import FastText as Fasttext
            model = Fasttext.load_fasttext_format(model_file)
        return model

    @overrides
    def infer(self, instance, mean=False, *args, **kwargs):
        """
        Embed data
        Args:
            instance: sentence or list of sentences
            mean: return list of embeddings or numpy.mean()

        Returns:
            Embedded sentence or list of embedded sentences
        """
        res = []
        if type(instance) is str:
            res = self._encode(instance, mean)

        elif type(instance) is list:
            for sentence in instance:
                embedded_tokens = self._encode(sentence, mean)
                res.append(embedded_tokens)

        return res

    def _encode(self, sentence: str, mean):
        tokens = sentence.split()
        embedded_tokens = []
        for t in tokens:
            try:
                emb = self.tok2emb[t]
            except KeyError:
                try:
                    emb = self.model[t][:self.dim]
                except KeyError:
                    emb = np.zeros(self.dim, dtype=np.float32)
                self.tok2emb[t] = emb
            embedded_tokens.append(emb)

        if mean:
            filtered = [et for et in embedded_tokens if np.any(et)]
            if filtered:
                return np.mean(filtered, axis=0)
            return np.zeros(self.dim, dtype=np.float32)

        return embedded_tokens
