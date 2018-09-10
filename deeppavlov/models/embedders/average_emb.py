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

import numpy as np

from overrides import overrides
from typing import List, Union

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


@register('avr_emb')
class AvrEmb(Component):
    """
    The class implements the functional of embedding the sentence, averaging the embedding of its tokens.
    If the attributes "tokenizer" and "vec" are None, then simple averaging is applied to the embedding of tokens,
    else averaging is weighted by tf-idf coefficients.

    Args:
        tokenizer: tokenizer ref
        vec: vectorizer ref

    Attributes:
        tokenizer: tokenizer class from pipeline, default None
        vec: vectorizer class from pipeline, default None
    """

    def __init__(self, **kwargs):
        """
        Initialize tokenizer and tf-idf vectorizer from **kwargs dictionary (from config).
        """
        self.tokenizer = kwargs.get('tokenizer', None)
        self.vec_ = kwargs.get('vectorizer', None)
        if self.vec_ is not None:
            self.vec = self.vec_.vectorizer
        else:
            self.vec = None

    @overrides
    def __call__(self, data: List[Union[list, np.ndarray]], text: List[List[str]] = None,
                 *args, **kwargs) -> Union[List, np.ndarray]:
        """
        Infer on the given data

        Args:
            data: list of tokenized and vectorized text samples
            text: text samples
            *args: additional arguments
            **kwargs: additional arguments

        Returns:
            for each sentence:
                np.vector
        """
        result = []
        if self.vec is None:
            if isinstance(data, list):
                return self.average(data, result)
            elif isinstance(data, np.ndarray):
                return np.array(self.average(data, result))
        else:
            if isinstance(data, list):
                return self.weigh_tfidf(data, text, result)
            elif isinstance(data, np.ndarray):
                return np.array(self.weigh_tfidf(data, text, result))

    @staticmethod
    def average(data, res):
        """
        Simple averaging of the tokens embeddings.

        Args:
            data:  list of tokenized and vectorized text samples
            res: list for sentence embeddings

        Returns:
             list of sentence embeddings
        """
        for x in data:
            if len(x) == 0:
                res.append(np.zeros((len(data[0][0]),)))
            else:
                res.append(np.average(np.array(x), axis=0))
        return res

    def weigh_tfidf(self, data, text, res):
        """
        Weighted averaging of the tokens embeddings by tf-idf coefficients.

        Args:
            data:  list of tokenized and vectorized text samples
            text: text samples
            res: list for sentence embeddings

        Returns:
             list of sentence embeddings
        """
        feature_names = self.vec.get_feature_names()
        x_test = self.vec.transform(text)
        text = self.tokenizer(text)

        for i in range(len(text)):
            info = {}
            feature_index = x_test[i, :].nonzero()[1]
            tfidf_scores = zip(feature_index, [x_test[i, x] for x in feature_index])

            for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
                info[w] = s

            if len(text[i]) == 0:
                res.append(np.zeros((len(data[0][0]),)))
            else:
                weights = np.array([info[w] if w in info.keys() else 0.05 for w in text[i]])
                matrix = np.array(data[i])
                res.append(np.dot(weights, matrix))
        return res


@register('sent_emb_vocab')
class SentEmb(Component):
    """
    The class implements the functional of embedding the sentence, averaging the embedding of its tokens.
    The embeddings of tokens is averaged over the coefficients tf-idf.

    Args:
        vocab_path: path to txt file with corpus counter vocab

    Attributes:
        counter_vocab: dict that contains words frequency
        corpus_len: a number of words in corpus
    """

    def __init__(self, vocab_path: str, sklearn: bool = False, use_pos: bool = False,
                 pos_vocab_path: Union[str, None] = None, **kwargs):
        """
        Initialize counter vocab from vocab_path.
        """
        self.sklearn = sklearn
        self.use_pos = use_pos

        self.vocab_path = expand_path(vocab_path)
        self.pos_vocab = self.load_pos_vocab(expand_path(pos_vocab_path))
        self.counter_vocab, self.min_count = self.load_counter_vocab(self.vocab_path)
        self.corpus_len = self.check_corpus_len(self.counter_vocab)

    @staticmethod
    def check_corpus_len(vocab):
        n = 0
        for key, val in vocab.items():
            n += int(val)
        return n

    @staticmethod
    def load_counter_vocab(load_path):
        counter_vocab = dict()
        with open(load_path, 'r') as f:
            lines = f.readlines()
            f.close()

        min_val = np.inf
        for line in lines:
            key, val = line[:-1].split('\t')
            counter_vocab[key] = val
            if int(val) < min_val:
                min_val = int(val)

        return counter_vocab, min_val

    @staticmethod
    def load_pos_vocab(load_path):
        pos_vocab = dict()
        with open(load_path, 'r') as f:
            lines = f.readlines()
            f.close()

        for line in lines:
            key, val = line[:-1].split(' ')  # "\t"
            pos_vocab[key] = val

        return pos_vocab

    @overrides
    def __call__(self, emb_batch: List[Union[list, np.ndarray]], tokens_batch: List[List[str]] = None,
                 pos_batch: List[List[str]] = None, *args, **kwargs) -> Union[List, np.ndarray]:
        """
        Infer on the given data. Weighted averaging of the tokens embeddings by tf-idf coefficients.

        Args:
            data: list of tokenized and vectorized text samples
            text: text samples
            *args: additional arguments
            **kwargs: additional arguments

        Returns:
            for each sentence:
                np.vector
        """
        result = []

        for i in range(len(tokens_batch)):
            if len(tokens_batch[i]) == 0:
                result.append(np.zeros((len(emb_batch[0][0]),)))
            else:
                weights = self.tf_idf_weights(tokens_batch[i])
                matrix = np.array(emb_batch[i])

                if self.use_pos:
                    if pos_batch is None:
                        raise ConfigError("In averaging component config parameter 'use_pos' is true,"
                                          " but pos_batch input is None")
                    else:
                        pos_w = self.pos_weights(pos_batch[i])
                        new_w = np.multiply(weights, pos_w)
                        result.append(np.dot(new_w, matrix))
                else:
                    result.append(np.dot(weights, matrix))

        return np.array(result)

    def pos_weights(self, pos_sent):
        weights = []
        for tag in pos_sent:
            w_ = self.pos_vocab.get(tag, 1.0)
            weights.append(float(w_))

        return np.array(weights)

    def tf_idf_weights(self, sent):
        threshold = self.min_count
        weights = []
        for word in sent:
            w = self.counter_vocab.get(word, None)
            if w is not None:
                if self.sklearn:
                    tf = int(w) / self.corpus_len
                    idf = np.log(float(self.corpus_len) / tf) + 1.0
                    weights.append(tf*idf)
                else:
                    weights.append(1.0 / (1.0 + np.log(int(w))))
            else:
                weights.append(1.0 / (1.0 + np.log(int(threshold))))

        return np.array(weights)
