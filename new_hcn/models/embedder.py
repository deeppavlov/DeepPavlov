"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from gensim.models import word2vec
import numpy as np
from pathlib import Path

from deeppavlov.common import paths
from deeppavlov.common.registry import register_model
from deeppavlov.models.trainable import Trainable
from deeppavlov.models.inferable import Inferable

import os
import copy
import re
import urllib.request
import fasttext


@register_model('fasttext')
class FasttextUtteranceEmbed(Inferable):

    _model_dir_path = ''
    _model_fpath = ''

    @property
    def _model_path(self):
        return Path(paths.USR_PATH).joinpath(model_dir_path, model_fpath)

    def __init__(self, model_dir_path, model_fpath, dim=None, fast=True):
        self._corpus_path = corpus_path
        self._model_dir_path = model_dir_path
        self._model_fpath = model_fpath
        self.tok2emb = {}
        self.fast = fast

        self.fasttext_model = None
        if not os.path.isfile(self._model_path):
            emb_path = os.environ.get('EMBEDDINGS_URL')
            if not emb_path:
                raise RuntimeError('\n:: <ERR> no fasttext model provided\n')
            try:
                print('Trying to download a pretrained fasttext model'
                      ' from the repository')
                url = urllib.parse.urljoin(emb_path, self._model_fpath)
                urllib.request.urlretrieve(url, self._model_path)
                print('Downloaded a fasttext model')
            except Exception as e:
                raise RuntimeError('Looks like the `EMBEDDINGS_URL` variable'
                                   ' is set incorrectly', e)
        self.model = fasttext.load_model(self._model_path)
        self.dim = dim or self.model.dim
        if self.dim > self.model.dim:
            raise RuntimeError("Embeddings are too short")

    def __getitem__(self, token):
        if self.fast:
            if token not in self.tok2emb:
                self.tok2emb[token] = self.model[token][:self.dim]
            return self.tok2emb[token]
        return self.model[token][:self.dim]

    def _encode(self, utterance):
        embs = [self.__getitem__(t) for t in utterance.split(' ')]
        if embs:
            return np.mean(embs, axis=0)
        return np.zeros(self.dim, dtype=np.float32)

    def _emb2str(self, vec):
        string = ' '.join([str(el) for el in vec])
        return string

    def infer(self, utterance):
        return self._encode(utterance)
