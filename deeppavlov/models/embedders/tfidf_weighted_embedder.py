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
from typing import List, Union, Optional, Tuple

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
        as a weighted by special coefficients average of tokens embeddings. \
        Coefficients can be taken from the given TFIDF-vectorizer in ``vectorizer`` or \
        calculated as TFIDF from counter vocabulary given in ``counter_vocab_path``.
        Also one can give in ``tags_vocab_path`` path to the vocabulary with weights of tags, \
        in this case batch with tags should be given as a second input in ``__call__`` method.

    Args:
        embedder: embedder instance
        tokenizer: tokenizer instance, should be able to detokenize sentence
        pad_zero: whether to pad samples or not
        mean: whether to return mean token embedding
        tags_vocab_path: optional path to vocabulary with tags weights
        vectorizer: vectorizer instance should be trained with ``analyzer="word"``
        counter_vocab_path: path to counter vocabulary
        idf_base_count: minimal idf value (less time occured are not counted)
        log_base: logarithm base for TFIDF-coefficient calculation froom counter vocabulary
        min_idf_weight: minimal idf weight

    Attributes:
        embedder: embedder instance
        tokenizer: tokenizer instance, should be able to detokenize sentence
        dim: dimension of embeddings
        pad_zero: whether to pad samples or not
        mean: whether to return mean token embedding
        tags_vocab: vocabulary with weigths for tags
        vectorizer: vectorizer instance
        counter_vocab_path: path to counter vocabulary
        counter_vocab: counter vocabulary
        idf_base_count: minimal idf value (less time occured are not counted)
        log_base: logarithm base for TFIDF-coefficient calculation froom counter vocabulary
        min_idf_weight: minimal idf weight
    """

    def __init__(self,
                 embedder: Component,
                 tokenizer: Component = None,
                 pad_zero: bool = False,
                 mean: bool = False,
                 tags_vocab_path: str = None,
                 vectorizer: Component = None,
                 counter_vocab_path: str = None,
                 idf_base_count: int = 100,
                 log_base: int = 10,
                 min_idf_weight=0.0, **kwargs) -> None:
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

        if vectorizer and counter_vocab_path:
            raise ConfigError("TfidfWeightedEmbedder got vectorizer and counter_vocab_path simultaneously."
                              " Remove one of them, please")
        elif vectorizer:
            self.vectorizer = vectorizer
            self.vocabulary = np.array(self.vectorizer.model.get_feature_names())
        elif counter_vocab_path:
            self.counter_vocab_path = expand_path(counter_vocab_path)
            self.counter_vocab, self.min_count = self.load_counter_vocab(self.vocab_path)
            self.idf_base_count = idf_base_count
            self.log_base = log_base
            self.min_idf_weight = min_idf_weight
        else:
            raise ConfigError("TfidfWeightedEmbedder did not get vectorizer or counter_vocab_path."
                              " Set one of them, please")

        if tags_vocab_path:
            self.tags_vocab = self.load_pos_vocab(expand_path(tags_vocab_path))
        else:
            self.tags_vocab = None

    @staticmethod
    def load_tags_vocab(load_path: str) -> dict:
        """
        Load tag vocabulary from the given path, each key of the vocabulary is a tag, \
            and the corresponding value of the item is a coefficient of words with such tags to be multiplied for.

        Args:
            load_path: path to the vocabulary to be load from

        Returns:
            vocabulary
        """
        tags_vocab = dict()
        with open(load_path, 'r') as f:
            lines = f.readlines()
            f.close()

        for line in lines:
            key, val = line[:-1].split(' ')  # "\t"
            tags_vocab[key] = val

        return tags_vocab

    @staticmethod
    def load_counter_vocab(load_path: str) -> Tuple[dict, int]:
        """
        Load counter vocabulary from the given path

        Args:
            load_path: path to the vocabulary to be load from

        Returns:
            vocabulary
        """
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
    def space_detokenizer(batch: List[List[str]]) -> List[str]:
        """
        Detokenizer by default. Linking tokens by space symbol

        Args:
            batch: batch of tokenized texts

        Returns:
            batch of detokenized texts
        """
        return [" ".join(tokens) for tokens in batch]

    @overrides
    def __call__(self, batch: List[List[str]], tags_batch: Optional[List[List[str]]] = None,
                 *args, **kwargs) -> List[Union[list, np.ndarray]]:
        """
        Infer on the given data

        Args:
            batch: tokenized text samples
            tags_batch: optional batch of corresponding tags
            *args: additional arguments
            **kwargs: additional arguments

        Returns:

        """

        if self.tags_vocab:
            if tags_batch is None:
                raise ConfigError("TfidfWeightedEmbedder got 'tags_vocab_path' but did not get batch woith tags.")
            batch = [self._tags_encode(sample, tags_sample) for sample, tags_sample in zip(batch, tags_batch)]
        else:
            if tags_batch:
                raise ConfigError("TfidfWeightedEmbedder got tags batch, but 'tags_vocab_path' is empty.")
            batch = [self._encode(sample) for sample in batch]

        if self.pad_zero:
            batch = zero_pad(batch)

        return batch

    def _encode(self, tokens: List[str]) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed one text sample

        Args:
            tokens: tokenized text sample

        Returns:
            list of embedded tokens or array of mean values
        """
        detokenized_sample = self.tokenizer([tokens])[0]  # str
        vectorized_sample = self.vectorizer([detokenized_sample])  # (voc_size,)

        if self.vectorizer:
            weights = np.array([vectorized_sample[0, np.where(self.vocabulary == token)[0][0]]
                                if len(np.where(self.vocabulary == token)[0]) else 0.
                                for token in tokens])
        else:
            weights = np.array([self.get_weight(max(self.counter_vocab.get(token, 0), self.idf_base_count))
                                for token in tokens])

        embedded_tokens = np.array(self.embedder([tokens]))[0, :, :]

        if self.mean:
            embedded_tokens = np.average(embedded_tokens, weights=weights, axis=0)
        else:
            embedded_tokens = np.array([weights[i] * embedded_tokens[i] for i in range(len(tokens))])

        return embedded_tokens

    def get_weight(self, count: int) -> float:
        """
        Calculate the weight corresponding to the given count

        Args:
            count: the number of occurences of particular token

        Returns:
            weight
        """
        log_count = np.log(count) / np.log(self.log_base)
        log_base_count = np.log(self.idf_base_count) / np.log(self.log_base)
        weight = max(1.0 / (1.0 + log_count - log_base_count), self.min_idf_weight)
        return weight

    def _tags_encode(self, tokens: List[str], tags: List[str]) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed one text sample

        Args:
            tokens: tokenized text sample
            pos: tokenized pos sample

        Returns:
            list of embedded tokens or array of mean values
        """

        embedded_tokens = np.array(self.embedder([tokens]))[0, :, :]

        tags_weights = np.array([self.tags_vocab.get(tag, 1.0) for tag in tags])

        detokenized_sample = self.tokenizer([tokens])[0]  # str
        vectorized_sample = self.vectorizer([detokenized_sample])  # (voc_size,)

        if self.vectorizer:
            weights = np.array([vectorized_sample[0, np.where(self.vocabulary == token)[0][0]]
                                if len(np.where(self.vocabulary == token)[0]) else 0.
                                for token in tokens])
        else:
            weights = np.array([self.get_weight(max(self.counter_vocab.get(token, 0), self.idf_base_count))
                                for token in tokens])

        weights = np.multiply(weights, tags_weights)

        if self.mean:
            embedded_tokens = np.average(embedded_tokens, weights=weights, axis=0)
        else:
            embedded_tokens = np.array([weights[i] * embedded_tokens[i] for i in range(len(tokens))])

        return embedded_tokens

    def destroy(self):
        pass
