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

from logging import getLogger
from typing import List, Generator, Any, Optional, Union, Tuple

# from nltk.corpus import stopwords
# STOPWORDS = stopwords.words('russian')
import pymorphy2
from nltk.tokenize.toktok import ToktokTokenizer

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.tokenizers.utils import detokenize, ngramize

logger = getLogger(__name__)


@register('ru_tokenizer')
class RussianTokenizer(Component):
    """Tokenize or lemmatize a list of documents for Russian language. Default models are
    :class:`ToktokTokenizer` tokenizer and :mod:`pymorphy2` lemmatizer.
    Return a list of tokens or lemmas for a whole document.
    If is called onto ``List[str]``, performs detokenizing procedure.

    Args:
        stopwords: a list of stopwords that should be ignored during tokenizing/lemmatizing
         and ngrams creation
        ngram_range: size of ngrams to create; only unigrams are returned by default
        lemmas: whether to perform lemmatizing or not
        lowercase: whether to perform lowercasing or not; is performed by default by :meth:`_tokenize`
         and :meth:`_lemmatize` methods
        alphas_only: whether to filter out non-alpha tokens; is performed by default by :meth:`_filter`
         method

    Attributes:
        stopwords: a list of stopwords that should be ignored during tokenizing/lemmatizing
         and ngrams creation
        tokenizer: an instance of :class:`ToktokTokenizer` tokenizer class
        lemmatizer: an instance of :class:`pymorphy2.MorphAnalyzer` lemmatizer class
        ngram_range: size of ngrams to create; only unigrams are returned by default
        lemmas: whether to perform lemmatizing or not
        lowercase: whether to perform lowercasing or not; is performed by default by :meth:`_tokenize`
         and :meth:`_lemmatize` methods
        alphas_only: whether to filter out non-alpha tokens; is performed by default by :meth:`_filter`
         method
         tok2morph: token-to-lemma cache

    """

    def __init__(self, stopwords: Optional[List[str]] = None, ngram_range: List[int] = None,
                 lemmas: bool = False, lowercase: Optional[bool] = None,
                 alphas_only: Optional[bool] = None, **kwargs):

        if ngram_range is None:
            ngram_range = [1, 1]
        self.stopwords = stopwords or []
        self.tokenizer = ToktokTokenizer()
        self.lemmatizer = pymorphy2.MorphAnalyzer()
        self.ngram_range = tuple(ngram_range)  # cast JSON array to tuple
        self.lemmas = lemmas
        self.lowercase = lowercase
        self.alphas_only = alphas_only
        self.tok2morph = {}

    def __call__(self, batch: Union[List[str], List[List[str]]]) -> \
            Union[List[List[str]], List[str]]:
        """Tokenize or detokenize strings, depends on the type structure of passed arguments.

        Args:
            batch: a batch of documents to perform tokenizing/lemmatizing;
             or a batch of lists of tokens/lemmas to perform detokenizing

        Returns:
            a batch of lists of tokens/lemmas; or a batch of detokenized strings

        Raises:
            TypeError: If the first element of ``batch`` is neither ``List``, nor ``str``.

        """
        if isinstance(batch[0], str):
            if self.lemmas:
                return list(self._lemmatize(batch))
            else:
                return list(self._tokenize(batch))
        if isinstance(batch[0], list):
            return [detokenize(doc) for doc in batch]
        raise TypeError(
            "StreamSpacyTokenizer.__call__() is not implemented for `{}`".format(type(batch[0])))

    def _tokenize(self, data: List[str], ngram_range: Tuple[int, int] = (1, 1), lowercase: bool = True) \
            -> Generator[List[str], Any, None]:
        """Tokenize a list of documents.

       Args:
           data: a list of documents to tokenize
           ngram_range: size of ngrams to create; only unigrams are returned by default
           lowercase: whether to perform lowercasing or not; is performed by default by
           :meth:`_tokenize` and :meth:`_lemmatize` methods

       Yields:
           list of lists of ngramized tokens or list of detokenized strings

        Returns:
            None

       """
        # DEBUG
        # size = len(data)
        _ngram_range = self.ngram_range or ngram_range

        if self.lowercase is None:
            _lowercase = lowercase
        else:
            _lowercase = self.lowercase

        for i, doc in enumerate(data):
            # DEBUG
            # logger.info("Tokenize doc {} from {}".format(i, size))
            tokens = self.tokenizer.tokenize(doc)
            if _lowercase:
                tokens = [t.lower() for t in tokens]
            filtered = self._filter(tokens)
            processed_doc = ngramize(filtered, ngram_range=_ngram_range)
            yield from processed_doc

    def _lemmatize(self, data: List[str], ngram_range: Tuple[int, int] = (1, 1)) -> \
            Generator[List[str], Any, None]:
        """Lemmatize a list of documents.

        Args:
            data: a list of documents to tokenize
            ngram_range: size of ngrams to create; only unigrams are returned by default

        Yields:
            list of lists of ngramized tokens or list of detokenized strings

        Returns:
            None

        """
        # DEBUG
        # size = len(data)
        _ngram_range = self.ngram_range or ngram_range

        tokenized_data = list(self._tokenize(data))

        for i, doc in enumerate(tokenized_data):
            # DEBUG
            # logger.info("Lemmatize doc {} from {}".format(i, size))
            lemmas = []
            for token in doc:
                try:
                    lemma = self.tok2morph[token]
                except KeyError:
                    lemma = self.lemmatizer.parse(token)[0].normal_form
                    self.tok2morph[token] = lemma
                lemmas.append(lemma)
            filtered = self._filter(lemmas)
            processed_doc = ngramize(filtered, ngram_range=_ngram_range)
            yield from processed_doc

    def _filter(self, items: List[str], alphas_only: bool = True) -> List[str]:
        """Filter a list of tokens/lemmas.

        Args:
            items: a list of tokens/lemmas to filter
            alphas_only: whether to filter out non-alpha tokens

        Returns:
            a list of filtered tokens/lemmas

        """
        if self.alphas_only is None:
            _alphas_only = alphas_only
        else:
            _alphas_only = self.alphas_only

        if _alphas_only:
            filter_fn = lambda x: x.isalpha() and not x.isspace() and x not in self.stopwords
        else:
            filter_fn = lambda x: not x.isspace() and x not in self.stopwords

        return list(filter(filter_fn, items))

    def set_stopwords(self, stopwords: List[str]) -> None:
        """Redefine a list of stopwords.

       Args:
           stopwords: a list of stopwords

       Returns:
           None

       """
        self.stopwords = stopwords
