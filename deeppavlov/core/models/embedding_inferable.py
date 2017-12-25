# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import copy
import numpy as np
import urllib.request
from gensim.models.fasttext import FastText
import numpy as np
from pathlib import Path

class EmbeddingInferableModel(object):

    def __init__(self, fname, emb_dict_name,  embedding_dim, embedding_url=None):
        """Initialize the class according to given parameters."""
        self.tok2emb = {}
        self.embedding_dim = embedding_dim
        self.load_items(emb_dict_name)

        if not fname:
            raise RuntimeError('No pretrained fasttext model provided')
        self.fasttext_model_file = fname

        if not Path(self.fasttext_model_file).is_file():
            emb_path = embedding_url
            if not emb_path:
                raise RuntimeError('No pretrained fasttext model provided')
            fname = Path(self.fasttext_model_file).name
            try:
                print('Trying to download a pretrained fasttext model from repository')
                url = urllib.parse.urljoin(emb_path, fname)
                urllib.request.urlretrieve(url, self.fasttext_model_file)
                print('Downloaded a fasttext model')
            except Exception as e:
                raise RuntimeError('Looks like the `EMBEDDINGS_URL` variable is set incorrectly', e)
        self.model = FastText.load(self.fasttext_model_file)

    def add_items(self, sentence_li):
        """
        Method adds new items to tok2emb dictionary from given text
        Args:
            sentence_li: list of sentences

        Returns:
            Nothing
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

    def save_items(self, fname):
        """
        Method saves dictionary tok2emb to file
        Args:
            fname: name of file without extension

        Returns:
            Nothing
        """
        fname += '.emb'
        f = open(fname, 'w')
        string = '\n'.join([el[0] + ' ' + self.emb2str(el[1]) for el in self.tok2emb.items()])
        f.write(string)
        f.close()
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

    def load_items(self, fname):
        """
        Method initializes dict of embeddings from file
        Args:
            fname: file name without extension

        Returns:
            Nothing
        """

        if fname is None or not Path(fname).is_file():
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
        return

    def infer(self, tokens):
        """
        Method returns list of embeddings
        Args:
            tokens: list of tokens (words, punctuation symbols)

        Returns:
            List of embeddings
        """
        self.add_items(tokens)
        self.model.build_vocab(tokens, update=True)
        embedded_tokens = []
        for tok in tokens:
            embedded_tokens.append(self.tok2emb.get(tok))
        return embedded_tokens
