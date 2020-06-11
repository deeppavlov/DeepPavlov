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
from typing import List, Dict, Tuple, Any, Iterator, Optional

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('ner_few_shot_iterator')
class NERFewShotIterator(DataLearningIterator):
    """Dataset iterator for simulating few-shot Named Entity Recognition setting.

    Args:
        data: list of (x, y) pairs for every data type in ``'train'``, ``'valid'`` and ``'test'``
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching
        target_tag: the tag of interest. For this tag the few-shot setting will be simulated
        filter_bi: whether to filter BIO markup or not
        n_train_samples: number of training samples in the few shot setting. The validation and the test sets will be
            the same
        remove_not_targets: whether to replace all non target tags with `O` tag or not.
    """

    def __init__(self,
                 data: Dict[str, List[Tuple[Any, Any]]],
                 seed: int = None,
                 shuffle: bool = True,
                 target_tag: str = None,
                 filter_bi: bool = True,
                 n_train_samples: int = 20,
                 remove_not_targets: bool = True,
                 *args, **kwargs) -> None:
        super(NERFewShotIterator, self).__init__(data=data, seed=seed, shuffle=shuffle)
        self.target_tag = target_tag
        self.filter_bi = filter_bi
        self.n_train_samples = n_train_samples
        self.remove_not_targets = remove_not_targets
        if self.target_tag is None:
            raise RuntimeError('You must provide a target tag to NERFewShotIterator!')

        self.n_samples = len(self.train)

        if self.remove_not_targets:
            self._remove_not_target_tags()

        if self.filter_bi:
            for key in self.data:
                for n, (x, y) in enumerate(self.data[key]):
                    self.data[key][n] = [x, [re.sub('(B-|I-)', '', tag) for tag in y]]

        self.tag_map = np.zeros(self.n_samples, dtype=bool)
        for n, (toks, tags) in enumerate(self.data['train']):
            if self.filter_bi:
                self.tag_map[n] = any(self.target_tag == tag for tag in tags if len(tag) > 2)
            else:
                self.tag_map[n] = any(self.target_tag == tag[2:] for tag in tags if len(tag) > 2)

        self.marked_nums = None
        self.unmarked_nums = None
        self._sample_marked()

    def _sample_marked(self):
        np.zeros(len(self.data['train']), dtype=bool)
        n_marked = 0
        self.marked_mask = np.zeros(self.n_samples, dtype=bool)
        while n_marked < self.n_train_samples:
            is_picked = True
            while is_picked:
                n = np.random.randint(self.n_samples)
                if not self.marked_mask[n]:
                    is_picked = False
                    self.marked_mask[n] = True
                    if self.tag_map[n]:
                        n_marked += 1

        self.marked_nums = np.arange(self.n_samples)[self.marked_mask]
        self.unmarked_nums = np.arange(self.n_samples)[~self.marked_mask]

    def _remove_not_target_tags(self):
        if self.remove_not_targets:
            for key in self.data:
                for n, (x, y) in enumerate(self.data[key]):
                    tags = []
                    for tag in y:
                        if tag.endswith('-' + self.target_tag):
                            tags.append(tag)
                        else:
                            tags.append('O')
                    self.data[key][n] = [x, tags]

    def get_instances(self, data_type: str = 'train') -> Tuple[List[List[str]], List[List[str]]]:
        """Get all data for a selected data type

        Args:
            data_type (str): can be either ``'train'``, ``'test'``, ``'valid'`` or ``'all'``

        Returns:
             a tuple of all inputs for a data type and all expected outputs for a data type
        """

        if data_type == 'train':
            samples = [self.data[data_type][i] for i in self.marked_nums]
        else:
            samples = self.data[data_type][:]

        x, y = list(zip(*samples))

        return x, y

    def gen_batches(self, batch_size: int,
                    data_type: str = 'train',
                    shuffle: Optional[bool] = None) -> Iterator[Tuple[List[List[str]], List[List[str]]]]:
        x, y = self.get_instances(data_type)
        data_len = len(x)

        if data_len == 0:
            return

        order = list(range(data_len))
        if shuffle is None and self.shuffle:
            self.random.shuffle(order)
        elif shuffle:
            self.random.shuffle(order)

        if batch_size < 0:
            batch_size = data_len

        for i in range((data_len - 1) // batch_size + 1):
            yield tuple(zip(*[(x[o], y[o]) for o in order[i * batch_size:(i + 1) * batch_size]]))
