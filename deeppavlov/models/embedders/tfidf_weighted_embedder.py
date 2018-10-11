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
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.data.utils import zero_pad

log = get_logger(__name__)


@register('tfidf_weighted_embedder')
class TfidfWeightedEmbedder(Component):
    """
    The class implements the functional of embedding the sentence \
        as a weighted by tf-idf coefficients sum of tokens embeddings.
    Args:
        embedder: embedder instance
        vectorizer: vectorizer instance should be trained with ``analyzer="word"``
        tokenizer: tokenizer instance, should be able to detokenize sentence
        pad_zero: whether to pad samples or not
        mean: whether to return mean token embedding
    Attributes:
        embedder: embedder instance
        vectorizer: vectorizer instance
        tokenizer: tokenizer instance, should be able to detokenize sentence
        dim: dimension of embeddings
        pad_zero: whether to pad samples or not
        mean: whether to return mean token embedding
    """

    def __init__(self,
                 embedder: Component,
                 vectorizer: Component,
                 tokenizer: Component = None,
                 pad_zero: bool = False,
                 mean: bool = False,
                 use_pos: bool=False,
                 pos_vocab_path: Union[str, None] = None,
                 **kwargs) -> None:
        """
        Initialize embedder with given parameters.
        """
        self.embedder = embedder
        self.dim = self.embedder.dim
        self.mean = mean
        self.pad_zero = pad_zero

        if tokenizer is None:
            self.tokenizer = self.space_detokenizer
        else:
            self.tokenizer = tokenizer

        self.vectorizer = vectorizer
        self.vocabulary = np.array(self.vectorizer.model.get_feature_names())

        self.use_pos = use_pos
        if self.use_pos:
            self.pos_vocab = self.load_pos_vocab(expand_path(pos_vocab_path))

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

    @staticmethod
    def space_detokenizer(batch: List[List[str]]) -> List[str]:
        return [" ".join(tokens) for tokens in batch]

    @overrides
    def __call__(self, batch: List[List[str]], pos_batch: Union[List[List[str]], None]=None,
                 *args, **kwargs) -> List[Union[list, np.ndarray]]:
        """
        Infer on the given data

        Args:
            batch: tokenized text samples
            *args: additional arguments
            **kwargs: additional arguments

        Returns:
        """

        if not self.use_pos:
            if pos_batch is not None:
                raise ConfigError("TfidfWeightedEmbedder get a POS batch, but 'use_pos' attribute is False. "
                                  "Check the config. (Default value of 'use_pos' is False)")
            batch = [self.encode(sample) for sample in batch]
        else:
            if pos_batch is None:
                raise ConfigError("TfidfWeightedEmbedder get a None instead a POS batch. "
                                  "Check the config. (Default value of 'use_pos' is False)")
            batch = [self.pos_encode(sample, pos_sample) for sample, pos_sample in zip(batch, pos_batch)]

        if self.pad_zero:
            batch = zero_pad(batch)

        return batch

    def encode(self, tokens: List[str]) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed one text sample

        Args:
            tokens: tokenized text sample

        Returns:
            list of embedded tokens or array of mean values
        """
        detokenized_sample = self.tokenizer([tokens])[0]  # str
        vectorized_sample = self.vectorizer([detokenized_sample])  # (voc_size,)

        weights = np.array([vectorized_sample[0, np.where(self.vocabulary == token)[0][0]]
                            if len(np.where(self.vocabulary == token)[0]) else 0.
                            for token in tokens])
        embedded_tokens = np.array(self.embedder([tokens]))[0, :, :]

        if self.mean:
            embedded_tokens = np.average(embedded_tokens, weights=weights, axis=0)
        else:
            embedded_tokens = np.array([weights[i] * embedded_tokens[i] for i in range(len(tokens))])

        return embedded_tokens

    def pos_encode(self, tokens: List[str], pos: List[str]) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed one text sample
        Args:
            tokens: tokenized text sample
            pos: tokenized pos sample
        Returns:
            list of embedded tokens or array of mean values
        """

        embedded_tokens = np.array(self.embedder([tokens]))[0, :, :]

        pos_weights = np.array([self.pos_vocab.get(tag, 1.0) for tag in pos])

        detokenized_sample = self.tokenizer([tokens])[0]  # str
        vectorized_sample = self.vectorizer([detokenized_sample])  # (voc_size,)

        tfidf_weights = np.array([vectorized_sample[0, np.where(self.vocabulary == token)[0][0]]
                            if len(np.where(self.vocabulary == token)[0]) else 0.
                            for token in tokens])

        weights = np.multiply(tfidf_weights, pos_weights)

        if self.mean:
            embedded_tokens = np.average(embedded_tokens, weights=weights, axis=0)
        else:
            embedded_tokens = np.array([weights[i] * embedded_tokens[i] for i in range(len(tokens))])

        return embedded_tokens

    def destroy(self):
        pass


@register('tfidf_vocab_weighted_embedder')
class TfidfVocabWeightedEmbedder(Component):
    """
    The class implements the functional of embedding the sentence \
        as a weighted by tf-idf coefficients sum of tokens embeddings.
    Args:
        embedder: embedder instance
        vectorizer: vectorizer instance should be trained with ``analyzer="word"``
        tokenizer: tokenizer instance, should be able to detokenize sentence
        pad_zero: whether to pad samples or not
        mean: whether to return mean token embedding
    Attributes:
        embedder: embedder instance
        vectorizer: vectorizer instance
        tokenizer: tokenizer instance, should be able to detokenize sentence
        dim: dimension of embeddings
        pad_zero: whether to pad samples or not
        mean: whether to return mean token embedding
    """

    def __init__(self,
                 embedder: Component,
                 vocab_path: str,
                 tokenizer: Component = None,
                 pad_zero: bool = False,
                 mean: bool = False,
                 use_pos: bool = False,
                 pos_vocab_path: Union[str, None] = None,
                 idf_base_count: int = 100,
                 log_base: int = 10,
                 min_idf_weight=0.0,
                 **kwargs) -> None:
        """
        Initialize embedder with given parameters.
        """
        self.embedder = embedder
        self.dim = self.embedder.dim
        self.mean = mean
        self.pad_zero = pad_zero
        self.idf_base_count = idf_base_count
        self.log_base = log_base
        self.min_idf_weight = min_idf_weight

        if tokenizer is None:
            self.tokenizer = self.space_detokenizer
        else:
            self.tokenizer = tokenizer

        self.vocab_path = expand_path(vocab_path)
        self.counter_vocab, self.min_count = self.load_counter_vocab(self.vocab_path)

        self.use_pos = use_pos
        if self.use_pos:
            self.pos_vocab = self.load_pos_vocab(expand_path(pos_vocab_path))

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

    @staticmethod
    def space_detokenizer(batch: List[List[str]]) -> List[str]:
        return [" ".join(tokens) for tokens in batch]

    @overrides
    def __call__(self, batch: List[List[str]], pos_batch: Union[List[List[str]], None]=None,
                 *args, **kwargs) -> List[Union[list, np.ndarray]]:
        """
        Infer on the given data
        Args:
            batch: tokenized text samples
            *args: additional arguments
            **kwargs: additional arguments
        Returns:
        """

        if not self.use_pos:
            if pos_batch is not None:
                raise ConfigError("TfidfWeightedEmbedder get a POS batch, but 'use_pos' attribute is False. "
                                  "Check the config. (Default value of 'use_pos' is False)")
            batch = [self.encode(sample) for sample in batch]
        else:
            if pos_batch is None:
                raise ConfigError("TfidfWeightedEmbedder get a None instead a POS batch. "
                                  "Check the config. (Default value of 'use_pos' is False)")
            batch = [self.pos_encode(sample, pos_sample) for sample, pos_sample in zip(batch, pos_batch)]

        if self.pad_zero:
            batch = zero_pad(batch)

        return batch

    def get_w(self, count):
        log_count = np.log(count) / np.log(self.log_base)
        log_base_count = np.log(self.idf_base_count) / np.log(self.log_base)
        weight = max(1.0 / (1.0 + log_count - log_base_count), self.min_idf_weight)
        return weight

    def encode(self, tokens: List[str]) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed one text sample
        Args:
            tokens: tokenized text sample
        Returns:
            list of embedded tokens or array of mean values
        """
        embedded_tokens = np.array(self.embedder([tokens]))[0, :, :]

        weights = np.array([self.get_w(max(self.counter_vocab.get(token, 0), self.idf_base_count)) for token in tokens])

        if self.mean:
            embedded_tokens = np.average(embedded_tokens, weights=weights, axis=0)
        else:
            embedded_tokens = np.array([weights[i] * embedded_tokens[i] for i in range(len(tokens))])

        return embedded_tokens

    def pos_encode(self, tokens: List[str], pos: List[str]) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed one text sample
        Args:
            tokens: tokenized text sample
            pos: tokenized pos sample
        Returns:
            list of embedded tokens or array of mean values
        """

        embedded_tokens = np.array(self.embedder([tokens]))[0, :, :]
        tfidf_weights = np.array(
            [self.get_w(max(self.counter_vocab.get(token, 0), self.idf_base_count)) for token in tokens])
        pos_weights = np.array([self.pos_vocab.get(tag, 1.0) for tag in pos])

        weights = np.multiply(tfidf_weights, pos_weights)

        if self.mean:
            embedded_tokens = np.average(embedded_tokens, weights=weights, axis=0)
        else:
            embedded_tokens = np.array([weights[i] * embedded_tokens[i] for i in range(len(tokens))])

        return embedded_tokens

    def destroy(self):
        pass
