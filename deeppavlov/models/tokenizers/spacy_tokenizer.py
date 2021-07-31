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
from typing import List, Generator, Any, Optional, Union, Tuple, Iterable

import spacy
import spacy.language

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.tokenizers.utils import detokenize, ngramize

logger = getLogger(__name__)


def _try_load_spacy_model(model_name: str, disable: Iterable[str] = ()):
    disable = set(disable)
    try:
        model = spacy.load(model_name, disable=disable)
    except OSError as e:
        try:
            model = __import__(model_name).load(disable=disable)
            if not isinstance(model, spacy.language.Language):
                raise RuntimeError(f'{model_name} is not a spacy model module')
        except Exception:
            raise e
    return model


@register('stream_spacy_tokenizer')
class StreamSpacyTokenizer(Component):
    """Tokenize or lemmatize a list of documents. Default spacy model is **en_core_web_sm**.
    Return a list of tokens or lemmas for a whole document.
    If is called onto ``List[str]``, performs detokenizing procedure.

    Args:
        disable: spacy pipeline elements to disable, serves a purpose of performing; if nothing
        stopwords: a list of stopwords that should be ignored during tokenizing/lemmatizing
         and ngrams creation
        batch_size: a batch size for spaCy buffering
        ngram_range: size of ngrams to create; only unigrams are returned by default
        lemmas: whether to perform lemmatizing or not
        lowercase: whether to perform lowercasing or not; is performed by default by :meth:`_tokenize`
         and :meth:`_lemmatize` methods
        alphas_only: whether to filter out non-alpha tokens; is performed by default by
         :meth:`_filter` method
        spacy_model: a string name of spacy model to use; DeepPavlov searches for this name in
         downloaded spacy models; default model is **en_core_web_sm**, it downloads automatically
         during DeepPavlov installation


    Attributes:
        stopwords: a list of stopwords that should be ignored during tokenizing/lemmatizing
         and ngrams creation
        model: a loaded spacy model
        batch_size: a batch size for spaCy buffering
        ngram_range: size of ngrams to create; only unigrams are returned by default
        lemmas: whether to perform lemmatizing or not
        lowercase: whether to perform lowercasing or not; is performed by default by :meth:`_tokenize`
         and :meth:`_lemmatize` methods
        alphas_only: whether to filter out non-alpha tokens; is performed by default by :meth:`_filter`
         method

    """

    def __init__(self, disable: Optional[Iterable[str]] = None, filter_stopwords: bool = False,
                 batch_size: Optional[int] = None, ngram_range: Optional[List[int]] = None,
                 lemmas: bool = False, lowercase: Optional[bool] = None, alphas_only: Optional[bool] = None,
                 spacy_model: str = 'en_core_web_sm', **kwargs):

        if disable is None:
            disable = ['parser', 'ner']
        if ngram_range is None:
            ngram_range = [1, 1]
        self.model = _try_load_spacy_model(spacy_model, disable=disable)
        self.filter_stopwords = filter_stopwords
        self.stopwords = spacy.lang.en.stop_words.STOP_WORDS if self.filter_stopwords else []
        self.batch_size = batch_size
        self.ngram_range = tuple(ngram_range)  # cast JSON array to tuple
        self.lemmas = lemmas
        self.lowercase = lowercase
        self.alphas_only = alphas_only

    def __call__(self, batch: Union[List[str], List[List[str]]]) -> Union[List[List[str]], List[str]]:
        """Tokenize or detokenize strings, depends on the type structure of passed arguments.

        Args:
            batch: a batch of documents to perform tokenizing/lemmatizing;
             or a batch of lists of tokens/lemmas to perform detokenizing

        Returns:
            a batch of lists of tokens/lemmas; or a batch of detokenized strings

        Raises:
            TypeError: If the first element of ``batch`` is neither List, nor str.

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

    def _tokenize(self, data: List[str], ngram_range: Optional[Tuple[int, int]] = None, batch_size: int = 10000,
                  lowercase: bool = True) -> Generator[List[str], Any, None]:
        """Tokenize a list of documents.

        Args:
            data: a list of documents to tokenize
            ngram_range: size of ngrams to create; only unigrams are returned by default
            batch_size: a batch size for spaCy buffering
            lowercase: whether to perform lowercasing or not; is performed by default by
                :meth:`_tokenize` and :meth:`_lemmatize` methods

        Yields:
            list of lists of ngramized tokens or list of detokenized strings

        Returns:
            None

        """
        _batch_size = self.batch_size or batch_size
        _ngram_range = ngram_range or self.ngram_range

        if self.lowercase is None:
            _lowercase = lowercase
        else:
            _lowercase = self.lowercase

        for i, doc in enumerate(
                self.model.tokenizer.pipe(data, batch_size=_batch_size)):
            if _lowercase:
                tokens = [t.lower_ for t in doc]
            else:
                tokens = [t.text for t in doc]
            filtered = self._filter(tokens)
            processed_doc = ngramize(filtered, ngram_range=_ngram_range, doc=data[i])
            yield from processed_doc

    def _lemmatize(self, data: List[str], ngram_range: Optional[Tuple[int, int]] = None, batch_size: int = 10000,
                   lowercase: bool = True) -> Generator[List[str], Any, None]:
        """Lemmatize a list of documents.

        Args:
            data: a list of documents to tokenize
            ngram_range: size of ngrams to create; only unigrams are returned by default
            batch_size: a batch size for spaCy buffering

       Yields:
           list of lists of ngramized lemmas or list of detokenized strings

        Returns:
            None

        """
        _batch_size = self.batch_size or batch_size
        _ngram_range = ngram_range or self.ngram_range

        if self.lowercase is None:
            _lowercase = lowercase
        else:
            _lowercase = self.lowercase

        for i, doc in enumerate(
                self.model.pipe(data, batch_size=_batch_size)):
            lemmas = [t.lemma_ for t in doc]
            if _lowercase:
                lemmas = [t.lower() for t in lemmas]
            filtered = self._filter(lemmas)
            processed_doc = ngramize(filtered, ngram_range=_ngram_range, doc=data[i])
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
