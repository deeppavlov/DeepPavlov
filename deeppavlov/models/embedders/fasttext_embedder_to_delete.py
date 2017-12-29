import os
from urllib import request, parse

import numpy as np
import fasttext
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.inferable import Inferable


@register('fasttext')
class FasttextEmbedderToDelete(Inferable):
    def __init__(self, model_dir='fasttext', dim=None, fast=True):
        self.tok2emb = {}
        self.fast = fast

        self.tok2emb = {}
        self.dim = dim
        self.embedding_url = embedding_url
        self.model_path = model_path
        self._model_dir = model_dir
        self._model_file = model_file
        self.model = self.load()

        self.fasttext_model = None
        if not self._model_path.is_file():
            emb_path = os.environ.get('EMBEDDINGS_URL')
            if not emb_path:
                raise RuntimeError('\n:: <ERR> no fasttext model provided\n')
            try:
                print('Trying to download a pretrained fasttext model'
                      ' from the repository')
                url = parse.urljoin(emb_path, self._model_fpath)
                request.urlretrieve(url, self._model_path.as_posix())
                print('Downloaded a fasttext model')
            except Exception as e:
                raise RuntimeError('Looks like the `EMBEDDINGS_URL` variable'
                                   ' is set incorrectly', e)
        print("Found fasttext model", self._model_path)
        self.model = fasttext.FastText(self._model_path.as_posix())
        self.dim = dim or self.model.args['dim']
        if self.dim > self.model.args['dim']:
            raise RuntimeError("Embeddings are too short")

    def __getitem__(self, token):
        if self.fast:
            if token not in self.tok2emb:
                self.tok2emb[token] = self.model[token][:self.dim]
            return self.tok2emb[token]
        return self.model.get_numpy_vector(token)[:self.dim]

    def _encode(self, sentence: str):
        if self.fast:
            embs = [self.__getitem__(t) for t in sentence.split()]
            if embs:
                return np.mean(embs, axis=0)
            return np.zeros(self.dim, dtype=np.float32)
        return self.model.get_numpy_sentence_vector(sentence)

    def _emb2str(self, vec):
        string = ' '.join([str(el) for el in vec])
        return string

    @overrides
    def infer(self, utterance, *args, **kwargs):
        return self._encode(utterance)
