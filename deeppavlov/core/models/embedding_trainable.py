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

from gensim.models.fasttext import FastText
import numpy as np


class EmbeddingTrainableModel(object):

    def __init__(self, embedding_dim, *args, **kwargs):
        self.embedding_dim = embedding_dim
        self.tok2emb = {}
        self.model = FastText(size=embedding_dim, min_count=1)

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

    def train(self, data, **kwargs):
        """
        Method trains fasttext model using text from the given file
        Args:
            filename: *.txt file containing text
            kwargs: additinal params for training

        Returns:
            Nothing
        """
        self.model.build_vocab(data)
        self.model.train(sentences=data, **kwargs)
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

    def save(self, fname=None, save_dict=True):
        """
        Method saves bin of fasttext model and dictionary of embeddings
        Args:
            fname: path to file without extension
            save_dict: flag to save embedding dictionary or not

        Returns:
            Nothing
        """
        if fname is None:
            raise IOError("No name for fasttext model is given")
        self.model.save(fname + ".bin")
        if save_dict:
            self.save_items(fname)
        return
