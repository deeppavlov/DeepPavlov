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

from typing import List, Callable

from nltk import sent_tokenize

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logger = get_logger(__name__)


@register('document_chunker')
class DocumentChunker(Component):
    """ Make chunks from a document. Keep sentences if needed.

    Args:
        sentencize_fn: a function for sentence segmentation
        keep_sentences: whether to tear up sentences between chunks or not
        tokens_limit: a number of tokens in a single chunk

    Attributes:
        keep_sentences: whether to tear up sentences between chunks or not
        tokens_limit: a number of tokens in a single chunk

    """

    def __init__(self, sentencize_fn: Callable = sent_tokenize, keep_sentences: bool = True,
                 tokens_limit: int = 500, *args, **kwargs):
        self._sentencize_fn = sentencize_fn
        self.keep_sentences = keep_sentences
        self.tokens_limit = tokens_limit

    def __call__(self, docs: List[str]) -> List[List[str]]:
        """ Make chunks from a batch of documents.

        Args:
            docs: a batch of documents

        Returns:
            chunks of docs

        """
        batch_chunks = []

        for doc in docs:
            doc_chunks = []
            if self.keep_sentences:
                sentences = sent_tokenize(doc)
                _len = 0
                keep = []
                for s in sentences:
                    _len += len(s.split())
                    if _len > self.tokens_limit:
                        if keep:
                            doc_chunks.append(' '.join(keep))
                            _len = 0
                            keep.clear()
                    keep.append(s)

                if keep:
                    doc_chunks.append(' '.join(keep))
                batch_chunks.append(doc_chunks)
            else:
                split_doc = doc.split()
                doc_chunks = [split_doc[i:i + self.tokens_limit] for i in range(0, len(split_doc), self.tokens_limit)]
                batch_chunks.append(doc_chunks)

        return batch_chunks
