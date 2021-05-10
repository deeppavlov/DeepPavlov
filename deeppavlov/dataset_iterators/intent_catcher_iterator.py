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

import re
from logging import getLogger
from typing import Tuple, List, Dict, Any, Iterator

from xeger import Xeger

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.dataset_readers.dto.rasa.nlu import Intents, IntentDesc

log = getLogger(__name__)


@register('intent_catcher_iterator')
class IntentCatcherIterator(DataLearningIterator):
    """
    Iterates over data for Intent Catcher training.
    A subclass of :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`.

    Args:
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching
        limit: Maximum number of phrases, that are generated from input regexps.

    """

    def __init__(self,
                 data: Dict[str, List[Tuple[Any, Any]]],
                 seed: int = None,
                 shuffle: bool = True,
                 limit: int = 10) -> None:
        self.limit = limit
        super().__init__(data, seed, shuffle)

    def gen_batches(self,
                    batch_size: int,
                    data_type: str = 'train',
                    shuffle: bool = None) -> Iterator[Tuple]:
        """Generate batches of inputs and expected output to train
        Intents Catcher

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching

        Returns:
            regexps used in the passed data_type, list of sentences generated
                from the original regexps, list of generated senteces' labels
        """

        if shuffle is None:
            shuffle = self.shuffle

        ic_file_content: Intents = self.data[data_type]["nlu_lines"]
        sentences, labels = [], []
        for intent in ic_file_content.intents:
            for intent_line in intent.lines:
                sentences.append(intent_line.text)
                labels.append(intent.title)

        assert len(sentences) == len(labels), \
            "Number of labels is not equal to the number of sentences"

        try:
            regexps = [re.compile(s) for s in sentences]
        except Exception as e:
            log.error(f"Some sentences are not a consitent regular expressions")
            raise e

        proto_entries_indices = list(range(len(sentences)))
        if shuffle:
            self.random.shuffle(proto_entries_indices)

        if batch_size < 0:
            batch_size = len(proto_entries_indices)

        xeger = Xeger(self.limit)

        regexps, generated_sentences, generated_labels = [], [], []
        generated_cnt = 0
        for proto_entry_ix in proto_entries_indices:
            sent, lab = sentences[proto_entry_ix], labels[proto_entry_ix]
            regex_ = re.compile(sent)

            gx = {xeger.xeger(sent) for _ in range(self.limit)}
            generated_sentences.extend(gx)
            generated_labels.extend([lab for _ in range(len(gx))])
            regexps.extend([regex_ for _ in range(len(gx))])

            if len(generated_sentences) == batch_size:
                # tuple(zip) below does [r1, r2, ..], [s1, s2, ..] -> ((r1, s1), (r2, s2), ..)
                yield tuple(zip(regexps, generated_sentences)), generated_labels
                generated_cnt += len(generated_sentences)
                regexps, generated_sentences, generated_labels = [], [], []

        if generated_sentences:
            yield tuple(zip(regexps, generated_sentences)), generated_labels
            generated_cnt += len(generated_sentences)
            regexps, generated_sentences, generated_labels = [], [], []

        log.info(f"Original number of samples: {len(sentences)}"
                 f", generated samples: {generated_cnt}")