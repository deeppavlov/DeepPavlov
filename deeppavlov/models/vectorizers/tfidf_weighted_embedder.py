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

    def __init__(self, embedder: Component, vectorizer: Component, tokenizer: Component = None,
                 pad_zero: bool = False, mean: bool = False, **kwargs) -> None:
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

    @staticmethod
    def space_detokenizer(batch: List[List[str]]) -> List[str]:
        return [" ".join(tokens) for tokens in batch]

    @overrides
    def __call__(self, batch: List[List[str]], *args, **kwargs) -> List[Union[list, np.ndarray]]:
        """
        Infer on the given data

        Args:
            batch: tokenized text samples
            *args: additional arguments
            **kwargs: additional arguments

        Returns:

        """
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

        weights = np.array([vectorized_sample[0, np.where(self.vocabulary == token)[0][0]]
                            if len(np.where(self.vocabulary == token)[0]) else 0.
                            for token in tokens])
        embedded_tokens = np.array(self.embedder([tokens]))[0, :, :]

        if self.mean:
            embedded_tokens = np.average(embedded_tokens, weights=weights, axis=0)
        else:
            embedded_tokens = np.array([weights[i] * embedded_tokens[i] for i in range(len(tokens))])

        return embedded_tokens

    def destroy(self):
        pass