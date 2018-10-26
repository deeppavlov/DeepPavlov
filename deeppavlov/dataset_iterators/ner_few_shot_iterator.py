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

from random import Random
from typing import List, Dict, Tuple, Any, Iterator
import numpy as np
import re
from copy import deepcopy
from collections import Counter
from itertools import chain

from deeppavlov.core.common.registry import register


@register('ner_few_shot_iterator')
class NERFewShotIterator:
    """Dataset iterator for learning models, e. g. neural networks.

    Args:
        data: list of (x, y) pairs for every data type in ``'train'``, ``'valid'`` and ``'test'``
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching

    Attributes:
        shuffle: whether to shuffle data during batching
        random: instance of ``Random`` initialized with a seed
    """
    def split(self, *args, **kwargs):
        pass

    def __init__(self, data: Dict[str, List[Tuple[Any, Any]]], seed: int = None, shuffle: bool = True,
                 *args, **kwargs) -> None:
        self.shuffle = shuffle

        self.random = Random(seed)

        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None) -> Iterator[Tuple[tuple, tuple]]:
        """Generate batches of inputs and expected output to train neural networks

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching

        Yields:
             a tuple of a batch of inputs and a batch of expected outputs
        """
        if shuffle is None:
            shuffle = self.shuffle

        data = self.data[data_type]
        data_len = len(data)

        if data_len == 0:
            return

        order = list(range(data_len))
        if shuffle:
            self.random.shuffle(order)

        if batch_size < 0:
            batch_size = data_len

        for i in range((data_len - 1) // batch_size + 1):
            yield tuple(zip(*[data[o] for o in order[i * batch_size:(i + 1) * batch_size]]))

    def get_instances(self, data_type: str = 'train') -> Tuple[tuple, tuple]:
        """Get all data for a selected data type

        Args:
            data_type (str): can be either ``'train'``, ``'test'``, ``'valid'`` or ``'all'``

        Returns:
             a tuple of all inputs for a data type and all expected outputs for a data type
        """
        data = self.data[data_type]
        return tuple(zip(*data))


class Corp:
    def __init__(self, data, target_tag, n_samples=20, filter_bi=False):
        self.filter_bi = filter_bi

        self.dataset = data
        if self.filter_bi:
            for key in self.dataset:
                for n, (x, y) in enumerate(self.dataset[key]):
                    self.dataset[key][n] = [x, [re.sub('(B-|I-)', '', tag) for tag in y]]
        self.n_samples = n_samples
        n_train_samples = len(self.dataset['train'])
        self.emb_dim = self.dataset['train'][0][0].shape[-1]
        self.target_tag = target_tag
        self.all_freqs = Counter(chain(*[tags for _, tags in self.dataset['train']]))
        self.tag_freqs = {key[2:]: value for key, value in self.all_freqs.items() if key[0] == 'B'}
        self.tag_map = np.zeros(n_train_samples, dtype=bool)
        for n, (toks, tags) in enumerate(self.dataset['train']):
            if self.filter_bi:
                self.tag_map[n] = any(target_tag == tag for tag in tags if len(tag) > 2)
            else:
                self.tag_map[n] = any(target_tag == tag[2:] for tag in tags if len(tag) > 2)
        if not any(self.tag_map):
            raise RuntimeError(f'No tag{target_tag} found!')

        self.marked_mask = None
        self.marked_nums = None
        self.unmarked_nums = None
        self.resample_marked()

        for k, (_, tags) in enumerate(self.dataset['train']):
            for n, tag in enumerate(tags):
                if not self.marked_mask[k] and self.target_tag in tag:
                    self.dataset['train'][k][1][n] = 'O'
        for _, tags in self.dataset['valid']:
            for n, tag in enumerate(tags):
                if self.marked_mask[k] and self.target_tag in tag:
                    self.dataset['valid'][k][1][n] = 'O'
        self.tag_vocab = SimpleVocabulary(unk_token='<UNK>',
                                          pad_with_zeros=True,
                                          save_path='model/tag.vocab',
                                          load_path='model/tag.vocab')
        self.tag_vocab.fit([tags for _, tags in dataset['train']])

    def resample_marked(self):
        np.zeros(len(self.dataset['train']), dtype=bool)
        n_marked = 0
        n_train_samples = len(n_train_samples)
        while n_marked < self.n_samples:
            is_picked = True
            while is_picked:
                n = np.random.randint(n_train_samples)
                if not self.marked_mask[n]:
                    is_picked = False
                    self.marked_mask[n] = True
                    if self.tag_map[n]:
                        n_marked += 1

        self.marked_nums = np.arange(n_train_samples)[self.marked_mask]
        self.unmarked_nums = np.arange(n_train_samples)[~self.marked_mask]

    def batch_gen(self, batch_size, marked_part=0.0, n_batches=-1, datatype='train'):
        n_marked = int(np.sum(np.random.choice([0, 1], size=batch_size, p=[1 - marked_part, marked_part])))
        n_unmarked = batch_size - n_marked
        if n_batches == -1:
            n_batches = int(len(self.dataset[datatype]) / batch_size)
        order = list(np.random.permutation(self.unmarked_nums))
        for n in range(n_batches):
            marked_nums = list(np.random.choice(self.marked_nums, size=n_marked))
            unmarked_nums = order[n * n_unmarked: (n + 1) * n_unmarked]
            samples = [self.dataset[datatype][i] for i in marked_nums + unmarked_nums]

            embs_batch, tags_batch = list(zip(*samples))
            max_len = max(len(tags) for tags in tags_batch)
            x = np.zeros([len(samples), max_len, self.emb_dim])
            for n, emb in enumerate(embs_batch):
                x[n, :len(emb)] = emb

            mask = np.zeros([len(tags_batch), max_len])
            for n, tags in enumerate(tags_batch):
                mask[n, :len(tags)] = 1
            y = self.tag_vocab(tags_batch)
            yield (x, mask), y

    def batch_gen_eval(self, batch_size, datatype='train'):
        n_batches = int((len(self.dataset[datatype]) + 1) / batch_size)
        for n in range(n_batches):
            samples = self.dataset[datatype][n * batch_size: (n + 1) * batch_size]
            embs_batch, tags_batch = list(zip(*samples))
            max_len = max(len(tags) for tags in tags_batch)
            x = np.zeros([len(samples), max_len, self.emb_dim])
            for n, emb in enumerate(embs_batch):
                x[n, :len(emb)] = emb

            mask = np.zeros([len(tags_batch), max_len])
            for n, tags in enumerate(tags_batch):
                mask[n, :len(tags)] = 1
            y = self.tag_vocab(tags_batch)
            yield (x, mask), y

    def get_marked_data(self, index_tags=False, datatype='train'):
        if datatype == 'train':
            samples = [self.dataset[datatype][i] for i in self.marked_nums]
        else:
            samples = self.dataset[datatype][:]

        embs_batch, tags_batch = list(zip(*samples))

        if index_tags:
            y = self.tag_vocab(tags_batch)
        else:
            y = tags_batch
        return list(zip(embs_batch, y))
