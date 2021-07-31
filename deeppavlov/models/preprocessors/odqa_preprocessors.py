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

from itertools import chain
from logging import getLogger
from typing import List, Callable, Union, Tuple, Optional

from nltk import sent_tokenize

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logger = getLogger(__name__)


@register('document_chunker')
class DocumentChunker(Component):
    """Make chunks from a document or a list of documents. Don't tear up sentences if needed.

    Args:
        sentencize_fn: a function for sentence segmentation
        keep_sentences: whether to tear up sentences between chunks or not
        tokens_limit: a number of tokens in a single chunk (usually this number corresponds to the squad model limit)
        flatten_result: whether to flatten the resulting list of lists of chunks
        paragraphs: whether to split document by paragrahs; if set to True, tokens_limit is ignored

    Attributes:
        keep_sentences: whether to tear up sentences between chunks or not
        tokens_limit: a number of tokens in a single chunk
        flatten_result: whether to flatten the resulting list of lists of chunks
        paragraphs: whether to split document by paragrahs; if set to True, tokens_limit is ignored

    """

    def __init__(self, sentencize_fn: Callable = sent_tokenize, keep_sentences: bool = True,
                 tokens_limit: int = 400, flatten_result: bool = False,
                 paragraphs: bool = False, number_of_paragraphs: int = -1, *args, **kwargs) -> None:
        self._sentencize_fn = sentencize_fn
        self.keep_sentences = keep_sentences
        self.tokens_limit = tokens_limit
        self.flatten_result = flatten_result
        self.paragraphs = paragraphs
        self.number_of_paragraphs = number_of_paragraphs

    def __call__(self, batch_docs: List[Union[str, List[str]]],
                 batch_docs_ids: Optional[List[Union[str, List[str]]]] = None) -> \
            Union[Tuple[Union[List[str], List[List[str]]], Union[List[str], List[List[str]]]],
                  Union[List[str], List[List[str]]]]:
        """Make chunks from a batch of documents. There can be several documents in each batch.
        Args:
            batch_docs: a batch of documents / a batch of lists of documents
            batch_docs_ids (optional) : a batch of documents ids / a batch of lists of documents ids
        Returns:
            chunks of docs, flattened or not and
            chunks of docs ids, flattened or not if batch_docs_ids were passed
        """

        result = []
        result_ids = []

        empty_docs_ids_flag = False

        if not batch_docs_ids:
            empty_docs_ids_flag = True

        if empty_docs_ids_flag:
            batch_docs_ids = [[[] for j in i] for i in batch_docs]

        for ids, docs in zip(batch_docs_ids, batch_docs):
            batch_chunks = []
            batch_chunks_ids = []
            if isinstance(docs, str):
                docs = [docs]
                ids = [ids]

            for id, doc in zip(ids, docs):
                if self.paragraphs:
                    split_doc = doc.split('\n\n')
                    split_doc = [sd.strip() for sd in split_doc]
                    split_doc = list(filter(lambda x: len(x) > 40, split_doc))
                    if self.number_of_paragraphs != -1:
                        split_doc = split_doc[:self.number_of_paragraphs]
                    batch_chunks.append(split_doc)
                    batch_chunks_ids.append([id] * len(split_doc))
                else:
                    doc_chunks = []
                    if self.keep_sentences:
                        sentences = sent_tokenize(doc)
                        n_tokens = 0
                        keep = []
                        for s in sentences:
                            n_tokens += len(s.split())
                            if n_tokens > self.tokens_limit:
                                if keep:
                                    doc_chunks.append(' '.join(keep))
                                    n_tokens = 0
                                    keep.clear()
                            keep.append(s)
                        if keep:
                            doc_chunks.append(' '.join(keep))
                        batch_chunks.append(doc_chunks)
                        batch_chunks_ids.append([id] * len(doc_chunks))
                    else:
                        split_doc = doc.split()
                        doc_chunks = [split_doc[i:i + self.tokens_limit] for i in
                                      range(0, len(split_doc), self.tokens_limit)]
                        batch_chunks.append(doc_chunks)
                        batch_chunks_ids.append([id] * len(doc_chunks))
            result.append(batch_chunks)
            result_ids.append(batch_chunks_ids)

        if self.flatten_result:
            if isinstance(result[0][0], list):
                for i in range(len(result)):
                    flattened = list(chain.from_iterable(result[i]))
                    flattened_ids = list(chain.from_iterable(result_ids[i]))
                    result[i] = flattened
                    result_ids[i] = flattened_ids

        if empty_docs_ids_flag:
            return result

        return result, result_ids


@register('string_multiplier')
class StringMultiplier(Component):
    """Make a list of strings from a provided string. A length of the resulting list equals a length
    of a provided reference argument.

    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, batch_s: List[str], ref: List[str]) -> List[List[str]]:
        """ Multiply each string in a provided batch of strings.

        Args:
            batch_s: a batch of strings to be multiplied
            ref: a reference to obtain a length of the resulting list

        Returns:
            a multiplied s as list

        """
        res = []
        for s, r in zip(batch_s, ref):
            res.append([s] * len(r))

        return res
