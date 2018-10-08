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
        pad_zero: whether to pad samples or not
        mean: whether to return mean token embedding

    Attributes:
        embedder: embedder instance
        vectorizer: vectorizer instance
        dim: dimension of embeddings
        pad_zero: whether to pad samples or not
        mean: whether to return mean token embedding
    """

    def __init__(self, embedder: Component, vectorizer: Component,
                 pad_zero: bool = False, mean: bool = False, **kwargs) -> None:
        """
        Initialize embedder with given parameters.
        """
        self.embedder = embedder
        self.dim = self.embedder.dim
        self.mean = mean
        self.pad_zero = pad_zero
        self.vectorizer = vectorizer
        self.vocabulary = self.vectorizer.model.get_feature_names()

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

    def weigh_tfidf(self, data, text):
        """
        Weighted averaging of the tokens embeddings by tf-idf coefficients.

        Args:
            data:  list of tokenized and vectorized text samples
            text: text samples
            res: list for sentence embeddings

        Returns:
             list of sentence embeddings
        """
        res = []
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

    def _encode(self, tokens: List[str]) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed one text sample

        Args:
            tokens: tokenized text sample

        Returns:
            list of embedded tokens or array of mean values
        """
        embedded_tokens = []
        for t in tokens:
            try:
                emb = self.tok2emb[t]
            except KeyError:
                try:
                    emb = self.model.get_word_vector(t)[:self.dim]
                except KeyError:
                    emb = np.zeros(self.dim, dtype=np.float32)
                self.tok2emb[t] = emb
            embedded_tokens.append(emb)

        if self.mean:
            filtered = [et for et in embedded_tokens if np.any(et)]
            if filtered:
                return np.mean(filtered, axis=0)
            return np.zeros(self.dim, dtype=np.float32)

        return embedded_tokens

    def destroy(self):
        pass