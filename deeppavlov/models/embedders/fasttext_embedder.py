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

import os
import copy
import urllib.request
import pyfasttext
import fasttext
import numpy as np

from pathlib import Path

from deeppavlov.core.common import paths
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.inferable import Inferable


# TODO: add URL of english embeddings
os.environ['EMBEDDINGS_URL'] = ''


class EmbeddingsDict(object):
    """EmbeddingsDict

    Class provides embeddings for fasttext model.

    Attributes:
        tok2emb: dictionary gives embedding vector for token (word)
        embedding_dim: dimension of embeddings
        opt: given parameters
        fasttext_model_file: file contains fasttext binary model
    """
    def __init__(self, opt, embedding_dim):
        """Initialize the class according to given parameters."""
        self.tok2emb = {}
        self.embedding_dim = embedding_dim
        self.opt = copy.deepcopy(opt)
        self.load_items()

        if not self.opt.get('fasttext_model'):
            raise RuntimeError('No pretrained fasttext model provided')
        self.fasttext_model_file = self.opt.get('fasttext_model')
        if not os.path.isfile(self.fasttext_model_file):
            emb_path = os.environ.get('EMBEDDINGS_URL')
            if not emb_path:
                raise RuntimeError('No pretrained fasttext model provided')
            fname = os.path.basename(self.fasttext_model_file)
            try:
                print('Trying to download a pretrained fasttext model from repository')
                url = urllib.parse.urljoin(emb_path, fname)
                urllib.request.urlretrieve(url, self.fasttext_model_file)
                print('Downloaded a fasttext model')
            except Exception as e:
                raise RuntimeError('Looks like the `EMBEDDINGS_URL` variable is set incorrectly', e)
        self.fasttext_model = fasttext.load_model(self.fasttext_model_file)

    def add_items(self, sentence_li):
        """Add new items to tok2emb dictionary from given text."""
        for sen in sentence_li:
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            for tok in tokens:
                if self.tok2emb.get(tok) is None:
                    self.tok2emb[tok] = self.fasttext_model[tok]

    def save_items(self, fname):
        """Save dictionary tok2emb to file."""
        if self.opt.get('fasttext_embeddings_dict') is not None:
            fname = self.opt['fasttext_embeddings_dict']
        else:
            fname += '.emb'
        f = open(fname, 'w')
        string = '\n'.join([el[0] + ' ' + self.emb2str(el[1]) for el in self.tok2emb.items()])
        f.write(string)
        f.close()

    def emb2str(self, vec):
        """Return string corresponding to the given embedding vectors"""
        string = ' '.join([str(el) for el in vec])
        return string

    def load_items(self):
        """Initialize embeddings from file."""
        fname = None
        if self.opt.get('fasttext_embeddings_dict') is not None:
            fname = self.opt['fasttext_embeddings_dict']
        elif self.opt.get('pretrained_model') is not None:
            fname = self.opt['pretrained_model']+'.emb'
        elif self.opt.get('model_file') is not None:
            fname = self.opt['model_file']+'.emb'

        if fname is None or not os.path.isfile(fname):
            print('There is no %s file provided. Initializing new dictionary.' % fname)
        else:
            print('Loading existing dictionary from %s.' % fname)
            with open(fname, 'r') as f:
                for line in f:
                    values = line.rsplit(sep=' ', maxsplit=self.embedding_dim)
                    assert(len(values) == self.embedding_dim + 1)
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.tok2emb[word] = coefs


@register('fasttext')
class FasttextUtteranceEmbed(Inferable):

    _model_dir_path = ''
    _model_fpath = ''

    @property
    def _model_path(self):
        p = Path(self._model_dir_path).joinpath(self._model_fpath)
        if not p.is_absolute():
            p = Path(paths.USR_PATH).joinpath(p)
        return p

    def __init__(self, model_dir_path, model_fpath, dim=None, fast=True):
        self._model_dir_path = model_dir_path
        self._model_fpath = model_fpath
        self.tok2emb = {}
        self.fast = fast

        self.fasttext_model = None
        if not self._model_path.is_file():
            emb_path = os.environ.get('EMBEDDINGS_URL')
            if not emb_path:
                raise RuntimeError('\n:: <ERR> no fasttext model provided\n')
            try:
                print('Trying to download a pretrained fasttext model'
                      ' from the repository')
                url = urllib.parse.urljoin(emb_path, self._model_fpath)
                urllib.request.urlretrieve(url, self._model_path.as_posix())
                print('Downloaded a fasttext model')
            except Exception as e:
                raise RuntimeError('Looks like the `EMBEDDINGS_URL` variable'
                                   ' is set incorrectly', e)
        print("Found fasttext model", self._model_path)
        self.model = pyfasttext.FastText(self._model_path.as_posix())
        self.dim = dim or self.model.args['dim']
        if self.dim > self.model.args['dim']:
            raise RuntimeError("Embeddings are too short")

    def __getitem__(self, token):
        if self.fast:
            if token not in self.tok2emb:
                self.tok2emb[token] = self.model[token][:self.dim]
            return self.tok2emb[token]
        return self.model.get_numpy_vector(token)[:self.dim]

    def _encode(self, utterance):
        if self.fast:
            embs = [self.__getitem__(t) for t in utterance.split(' ')]
            if embs:
                return np.mean(embs, axis=0)
            return np.zeros(self.dim, dtype=np.float32)
        return model.get_numpy_sentence_vector(utterance)
        #return model.get_numpy_text_vector(utterance)

    def _emb2str(self, vec):
        string = ' '.join([str(el) for el in vec])
        return string

    def infer(self, utterance):
        return self._encode(utterance)
