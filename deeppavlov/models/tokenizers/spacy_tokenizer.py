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

from itertools import chain
from typing import List, Generator, Any, Tuple

import spacy
from spacy.lang.en import English
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.models.tokenizers.utils import detokenize, ngramize
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)


@register('stream_spacy_tokenizer')
class StreamSpacyTokenizer(Component):
    """
    Tokenize or lemmatize a list of documents.
    Return list of tokens or lemmas, without sentencizing.
    Works only for English language.
    """

    def __init__(self, disable: list = None, stopwords: list = None,
                 batch_size: int = None, ngram_range: List[int] = None, lemmas=False,
                 n_threads: int = None, lowercase: bool = None, alphas_only: bool = None, **kwargs):
        """
        :param disable: pipeline processors to omit; if nothing should be disabled,
         pass an empty list
        :param stopwords: a set of words to skip
        :param batch_size: a batch size for internal spaCy multi-threading
        :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
        [1, 2], for bigrams only should be set to [2, 2]
        :param lemmas: weather to perform lemmatizing or not while tokenizing, currently works only
        for the English language
        :param n_threads: a number of threads for internal spaCy multi-threading
        """
        if disable is None:
            disable = ['parser', 'ner']
        if ngram_range is None:
            ngram_range = [1, 1]
        self.stopwords = stopwords or []
        self.model = spacy.load('en', disable=disable)
        self.model.add_pipe(self.model.create_pipe('sentencizer'))
        self.tokenizer = English().Defaults.create_tokenizer(self.model)
        self.batch_size = batch_size
        self.ngram_range = tuple(ngram_range)  # cast JSON array to tuple
        self.lemmas = lemmas
        self.n_threads = n_threads
        self.lowercase = lowercase
        self.alphas_only = alphas_only

    def __call__(self, batch):
        if isinstance(batch[0], str):
            if self.lemmas:
                return list(self._lemmatize(batch))
            else:
                return list(self._tokenize(batch))
        if isinstance(batch[0], list):
            return [detokenize(doc) for doc in batch]
        raise TypeError(
            "StreamSpacyTokenizer.__call__() is not implemented for `{}`".format(type(batch[0])))

    def _tokenize(self, data: List[str], ngram_range=(1, 1), batch_size=10000, n_threads=1,
                  lowercase=True) -> Generator[List[str], Any, None]:
        """
        Tokenize a list of documents.
        :param data: a list of documents to process
        :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
        (1, 2), for bigrams only should be set to (2, 2)
        :param batch_size: the number of documents to process at once;
        improves the spacy 'pipe' performance; shouldn't be too small
        :param n_threads: a number of threads for parallel computing; doesn't work good
         on a standard Python
        :param lowercase: whether to perform lowercasing or not
        :return: a single processed doc generator
        """
        # DEBUG
        # size = len(data)
        _batch_size = self.batch_size or batch_size
        _ngram_range = self.ngram_range or ngram_range
        _n_threads = self.n_threads or n_threads

        if self.lowercase is None:
            _lowercase = lowercase
        else:
            _lowercase = self.lowercase

        # print(_lowercase)

        for i, doc in enumerate(
                self.tokenizer.pipe(data, batch_size=_batch_size, n_threads=_n_threads)):
            # DEBUG
            # logger.info("Tokenize doc {} from {}".format(i, size))
            if _lowercase:
                tokens = [t.lower_ for t in doc]
            else:
                tokens = [t.text for t in doc]
            filtered = self._filter(tokens)
            processed_doc = ngramize(filtered, ngram_range=_ngram_range)
            yield from processed_doc

    def _lemmatize(self, data: List[str], ngram_range=(1, 1), batch_size=10000, n_threads=1) -> \
            Generator[List[str], Any, None]:
        """
        Lemmatize a list of documents.
        :param data: a list of documents to process
        :param ngram_range: range for producing ngrams, ex. for unigrams + bigrams should be set to
        (1, 2), for bigrams only should be set to (2, 2)
        :param batch_size: the number of documents to process at once;
        improves the spacy 'pipe' performance; shouldn't be too small
        :param n_threads: a number of threads for parallel computing; doesn't work good
         on a standard Python
        :return: a single processed doc generator
        """
        # DEBUG
        # size = len(data)
        _batch_size = self.batch_size or batch_size
        _ngram_range = self.ngram_range or ngram_range
        _n_threads = self.n_threads or n_threads

        for i, doc in enumerate(
                self.model.pipe(data, batch_size=_batch_size, n_threads=_n_threads)):
            # DEBUG
            # logger.info("Lemmatize doc {} from {}".format(i, size))
            lemmas = chain.from_iterable([sent.lemma_.split() for sent in doc.sents])
            filtered = self._filter(lemmas)
            processed_doc = ngramize(filtered, ngram_range=_ngram_range)
            yield from processed_doc

    def _filter(self, items, alphas_only=True):
        """
        Make ngrams from a list of tokens/lemmas
        :param items: list of tokens, lemmas or other strings to form ngrams
        :param alphas_only: should filter numeric and alpha-numeric types or not
        :return: filtered list of tokens/lemmas
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

    def set_stopwords(self, stopwords):
        self.stopwords = stopwords
