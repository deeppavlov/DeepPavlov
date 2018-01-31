"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed inder the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import urllib
from pathlib import Path
from warnings import warn

import numpy as np
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.errors import ConfigError


@register('fasttext')
class FasttextEmbedder(Inferable):
    def __init__(self, save_path, load_path=None, dim=100,
                 embedding_url=None, emb_module='fasttext', **kwargs):
        """
        Args:
            ser_path: path to binary file with embeddings
            dim: dimension of embeddings
            embedding_url: url link to embedding to try to download if file does not exist
        """
        super().__init__(save_path=save_path,
                         load_path=load_path)
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

        if self.load_path:
            if self.load_path.is_file():
                print("[loading embeddings from `{}`]".format(self.load_path))
                model_file = str(self.load_path)
                if self.emb_module == 'fasttext':
                    import fastText as Fasttext
                    # model = Fasttext.load_model(model_file)
                    model = Fasttext.load_model(model_file)
                elif self.emb_module == 'pyfasttext':
                    from pyfasttext import FastText as Fasttext
                    model = Fasttext(model_file)
                else:
                    from gensim.models.wrappers.fasttext import FastText as Fasttext
                    model = Fasttext.load_fasttext_format(model_file)
            elif isinstance(self.load_path, Path):
                raise ConfigError("Provided `load_path` for {} doesn't exist!".format(
                    self.__class__.__name__))
        else:
            warn("No `load_path` is provided for {}".format(self.__class__.__name__))
            if self.embedding_url:
                try:
                    print('[trying to download a pretrained fasttext model from repository]')
                    local_filename, _ = urllib.request.urlretrieve(self.embedding_url)
                    with open(local_filename, 'rb') as fin:
                        model_file = fin.read()

                    mp = self.save_path
                    self.load_path = self.save_path
                    model = self.load()
                    print("[saving downloaded fasttext model to {}]".format(mp))
                    with open(str(mp), 'wb') as fout:
                        fout.write(model_file)
                except Exception as e:
                    raise RuntimeError(
                        'Looks like the provided fasttext url is incorrect', e)
            else:
                raise FileNotFoundError(
                    'No pretrained fasttext model provided or provided "load_path" is incorrect.'
                    ' Please include "load_path" to json.')

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
                    if self.emb_module == 'fasttext':
                        import fastText as Fasttext
                        emb = self.model.get_word_vector(t)[:self.dim]
                    else:
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
