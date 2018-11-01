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


from typing import List, Tuple, Iterator, Optional

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)

@register('elmo_file_paths_iterator')
class ELMoFilePathsIterator(DataLearningIterator):
    # """Dataset iterator for multiparagraph-SQuAD dataset.

    # With ``with_answer_rate`` rate samples context with answer and with ``1 - with_answer_rate`` samples context
    # from the same article, but without an answer. Contexts without an answer are sampled according to
    # their tfidf scores (tfidf score between question and context).

    # It extracts ``context``, ``question``, ``answer_text`` and ``answer_start`` position from dataset.
    # Example from a dataset is a tuple of ``(context, question)`` and ``(answer_text, answer_start)``. If there is
    # no answer in context, then ``answer_text`` is empty string and `answer_start` is equal to -1.

    # Args:
    #     data: dict with keys ``'train'``, ``'valid'`` and ``'test'`` and values
    #     seed: random seed for data shuffling
    #     shuffle: whether to shuffle data during batching
    #     with_answer_rate: sampling rate of contexts with answer

    # Attributes:
    #     shuffle: whether to shuffle data during batching
    #     random: instance of ``Random`` initialized with a seed
    # """

    def __init__(self, data, seed: Optional[int] = None, shuffle: bool = True,
                 *args, **kwargs) -> None:
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        super().__init__(data, seed, shuffle, *args, **kwargs)

    @staticmethod
    def _chunk_generator(items_list, chunk_size):
        for i in range(0, len(items_list), chunk_size):
            yield items_list[i:i + chunk_size]

    @staticmethod
    def _shard_generator(shards, shuffle = False, random = None):
        shards_to_choose = list(shards)
        if shuffle:
            random.shuffle(shards_to_choose)
        for shard in shards_to_choose:
            lines = open(shard).readlines()
            if shuffle:
                random.shuffle(lines)
            yield lines

    def gen_batches(self, batch_size: int, data_type: str = 'train', shuffle: bool = None)\
            -> Iterator[Tuple[str,str]]:
        if shuffle is None:
            shuffle = self.shuffle

        # import pdb; pdb.set_trace()
        tgt_data = self.data[data_type]
        log.info(data_type)
        shard_gen = self._shard_generator(tgt_data, shuffle = False, random = self.np_random)

        data = []
        bs = batch_size
        for shard in shard_gen:
            if not batch_size:
                bs = len(shard)
            data.extend(shard)
            chunk_gen = self._chunk_generator(data, bs)

            for lines in chunk_gen:
                if len(lines) < bs:
                    data = lines
                    break
                # print(tuple(lines))
                # out_lines = [lines[:5], lines[:5]]
                # print(len(lines))
                batch = [lines, lines]
                # input()
                yield batch


    def get_instances(self, data_type: str = 'train') -> Tuple[Tuple[Tuple[str, str]], Tuple[List[str], List[int]]]:
        data_examples = []
        for qcas in self.data[data_type]:  # question, contexts, answers
            question = qcas['question']
            for context in qcas['contexts']:
                answer_text = [x['text'] for x in context['answer']]
                answer_start = [x['answer_start'] for x in context['answer']]
                data_examples.append(((context['context'], question), (answer_text, answer_start)))
        return tuple(zip(*data_examples))


















# dataset = {'train': [
#     '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.json',
#     '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.json',
#     '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.json',
#            '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.1.json',
#            '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.1.json',
#            '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.1.json',
#            '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.1.json',
#            ]}

# list(shard_generator(dataset['train']))

#     def gen_batches(self, batch_size: int, data_type: str = 'train', shuffle: bool = None)\
#             -> Generator[Tuple[str]]:
#             # -> Generator[Tuple[Tuple[Tuple[str, str]], Tuple[List[str], List[int]]], None, None]:

#         # if shuffle is None:
#         #     shuffle = self.shuffle

#         # if data_type == 'train':
#         #     random = self.np_random
#         # else:
#         #     random = np.random.RandomState(self.seed)

#         tgt_data = self.data[data_type].copy()
#         if shuffle:
#             random.shuffle(tgt_data)
#         shard_gen = shard_generator(tgt_data, shuffle = False)

#         data = []
#         bs = batch_size
#         for shard in shard_gen:
#             if not batch_size:
#                 bs = len(shard)
#             data.extend(shard)
#             chunk_gen = self._chunk_generator(data, bs)

#             for batch in chunk_gen:
#                 if len(batch) < bs:
#                     data = batch
#                     break
#                 yield tuple(batch)


#     def _chunk_generator(self, items_list, chunk_size):
#         for i in range(0, len(items_list), chunk_size):
#                         yield items_list[i:i + chunk_size]

    

#     def get_instances(self, data_type: str = 'train') -> Tuple[Tuple[Tuple[str, str]], Tuple[List[str], List[int]]]:
#         data_examples = []
#         for qcas in self.data[data_type]:  # question, contexts, answers
#             question = qcas['question']
#             for context in qcas['contexts']:
#                 answer_text = [x['text'] for x in context['answer']]
#                 answer_start = [x['answer_start'] for x in context['answer']]
#                 data_examples.append(((context['context'], question), (answer_text, answer_start)))
#         return tuple(zip(*data_examples))

# # def _get_batch(generator, batch_size, num_steps, max_word_length):
# #     """Read batches of input."""
# #     cur_stream = [None] * batch_size

# #     no_more_data = False
# #     while True:
# #         inputs = np.zeros([batch_size, num_steps], np.int32)
# #         if max_word_length is not None:
# #             char_inputs = np.zeros([batch_size, num_steps, max_word_length],
# #                                 np.int32)
# #         else:
# #             char_inputs = None
# #         targets = np.zeros([batch_size, num_steps], np.int32)

# #         for i in range(batch_size):
# #             cur_pos = 0

# #             while cur_pos < num_steps:
# #                 if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
# #                     try:
# #                         cur_stream[i] = list(next(generator))
# #                     except StopIteration:
# #                         # No more data, exhaust current streams and quit
# #                         no_more_data = True
# #                         break

# #                 how_many = min(len(cur_stream[i][0]) - 1, num_steps - cur_pos)
# #                 next_pos = cur_pos + how_many

# #                 inputs[i, cur_pos:next_pos] = cur_stream[i][0][:how_many]
# #                 if max_word_length is not None:
# #                     char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][
# #                                                                     :how_many]
# #                 targets[i, cur_pos:next_pos] = cur_stream[i][0][1:how_many+1]

# #                 cur_pos = next_pos

# #                 cur_stream[i][0] = cur_stream[i][0][how_many:]
# #                 if max_word_length is not None:
# #                     cur_stream[i][1] = cur_stream[i][1][how_many:]

# #         if no_more_data:
# #             # There is no more data.  Note: this will not return data
# #             # for the incomplete batch
# #             break

# #         X = {'token_ids': inputs, 'tokens_characters': char_inputs,
# #                  'next_token_id': targets}

# #         yield X
        
# # class LMDataset(object):
# #     """
# #     Hold a language model dataset.

# #     A dataset is a list of tokenized files.  Each file contains one sentence
# #         per line.  Each sentence is pre-tokenized and white space joined.
# #     """
# #     def __init__(self, filepattern, vocab, reverse=False, test=False,
# #                  shuffle_on_load=False, get_aug_shard_name = None):
# #         '''
# #         filepattern = a glob string that specifies the list of files.
# #         vocab = an instance of Vocabulary or UnicodeCharsVocabulary
# #         reverse = if True, then iterate over tokens in each sentence in reverse
# #         test = if True, then iterate through all data once then stop.
# #             Otherwise, iterate forever.
# #         shuffle_on_load = if True, then shuffle the sentences after loading.
# #         '''
# #         self._vocab = vocab
# #         self._all_shards = glob.glob(filepattern)
# #         print('Found %d shards at %s' % (len(self._all_shards), filepattern))
# #         self._shards_to_choose = []

# #         self._reverse = reverse
# #         self._test = test
# #         self._shuffle_on_load = shuffle_on_load
# #         self._get_aug_shard_name = get_aug_shard_name
# #         self._use_char_inputs = hasattr(vocab, 'encode_chars')

# #         self._ids = self._load_random_shard()

# #     def _choose_random_shard(self):
# #         if len(self._shards_to_choose) == 0:
# #             self._shards_to_choose = list(self._all_shards)
# #             random.shuffle(self._shards_to_choose)
# #         shard_name = self._shards_to_choose.pop()
# #         return shard_name

# #     def _load_random_shard(self):
# #         """Randomly select a file and read it."""
# #         if self._test:
# #             if len(self._all_shards) == 0:
# #                 # we've loaded all the data
# #                 # this will propogate up to the generator in get_batch
# #                 # and stop iterating
# #                 raise StopIteration
# #             else:
# #                 shard_name = self._all_shards.pop()
# #         else:
# #             # just pick a random shard
# #             shard_name = self._choose_random_shard()

# #         ids = self._load_shard(shard_name)
# #         self._i = 0
# #         self._nids = len(ids)
# #         return ids

# #     def _reverse_sentence(self, sentences_raw):
# #         sentences = []
# #         for sentence in sentences_raw:
# #             splitted = sentence.split()
# #             splitted.reverse()
# #             sentences.append(' '.join(splitted))
# #         return sentences

# #     def _load_shard(self, shard_name):
# #         """Read one file and convert to ids.

# #         Args:
# #             shard_name: file path.

# #         Returns:
# #             list of (id, char_id) tuples.
# #         """
# #         def reverse_sentence(sentences_raw):
# #             sentences = []
# #             for sentence in sentences_raw:
# #                 splitted = sentence.split()
# #                 splitted.reverse()
# #                 sentences.append(' '.join(splitted))
# #             return sentences

# #         def get_sentence(shard_name):
# #             print('Loading data from: %s' % shard_name)
# #             with open(shard_name) as f:
# #                 sentences_raw = f.readlines()

# #             if self._reverse:
# #                 sentences = reverse_sentence(sentences_raw)
# #             else:
# #                 sentences = sentences_raw
# #             return sentences

# #         if self._get_aug_shard_name:
# #             features_shard_name, labels_shard_name = self._get_aug_shard_name(shard_name)
# #             features_sentences = get_sentence(features_shard_name)
# #             labels_sentences = get_sentence(labels_shard_name)
# #             sentences = list(zip(features_sentences, labels_sentences))
# #         else:
# #             sentences = get_sentence(shard_name)

# #         if self._shuffle_on_load:
# #             random.shuffle(sentences)

# #         if self._get_aug_shard_name:
# #             ids = [self.vocab.encode(sentence[1], self._reverse)
# #                    for sentence in sentences]
# #             if self._use_char_inputs:
# #                 chars_ids = [self.vocab.encode_chars(sentence[0], self._reverse)
# #                          for sentence in sentences]
# #             else:
# #                 chars_ids = [None] * len(ids)
# #         else:
# #             ids = [self.vocab.encode(sentence, self._reverse)
# #                    for sentence in sentences]
# #             if self._use_char_inputs:
# #                 chars_ids = [self.vocab.encode_chars(sentence, self._reverse)
# #                          for sentence in sentences]
# #             else:
# #                 chars_ids = [None] * len(ids)

# #         print('Loaded %d sentences.' % len(ids))
# #         print('Finished loading')
# #         return list(zip(ids, chars_ids))

# #     def get_sentence(self):
# #         while True:
# #             if self._i == self._nids:
# #                 self._ids = self._load_random_shard()
# #             ret = self._ids[self._i]
# #             self._i += 1
# #             yield ret

# #     @property
# #     def max_word_length(self):
# #         if self._use_char_inputs:
# #             return self._vocab.max_word_length
# #         else:
# #             return None

# #     def iter_batches(self, batch_size, num_steps):
# #         for X in _get_batch(self.get_sentence(), batch_size, num_steps,
# #                            self.max_word_length):

# #             # token_ids = (batch_size, num_steps)
# #             # char_inputs = (batch_size, num_steps, 50) of character ids
# #             # targets = word ID of next word (batch_size, num_steps)
# #             yield X

# #     @property
# #     def vocab(self):
# #         return self._vocab


# #%%
# import random
# #%%


# def shard_generator(shards, shuffle = False):
#     shards_to_choose = list(shards)
#     if shuffle:
#         random.shuffle(shards_to_choose)
#     for shard in shards_to_choose:
#         lines = open(shard).readlines()
#         if shuffle:
#             random.shuffle(lines)
#         yield lines
# dataset = {'train': [
#     '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.json',
#     '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.json',
#     '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.json',
#            '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.1.json',
#            '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.1.json',
#            '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.1.json',
#            '/home/den/Documents/elmo/elmo_emb2deeppavlov/work-dp/download/elmo/options.1.json',
#            ]}

# list(shard_generator(dataset['train']))

# def gen_batches(self, batch_size: int, data_type: str = 'train', shuffle: bool = None)\
#         -> Generator[Tuple[Tuple[Tuple[str, str]], Tuple[List[str], List[int]]], None, None]:

#     # if shuffle is None:
#     #     shuffle = self.shuffle

#     # if data_type == 'train':
#     #     random = self.np_random
#     # else:
#     #     random = np.random.RandomState(self.seed)

#     tgt_data = self.data[data_type].copy()
#     if shuffle:
#         random.shuffle(tgt_data)
#     shard_gen = shard_generator(tgt_data, shuffle = False)

#     data = []
#     bs = batch_size
#     for shard in shard_gen:
#         if not batch_size:
#             bs = len(shard)
#         data.extend(shard)
#         chunk_gen = _chunk_generator(data, bs)

#         for batch in chunk_gen:
#             if len(batch) < bs:
#                 data = batch
#                 break
#             yield tuple(batch)


# def _chunk_generator(items_list, chunk_size):
#     for i in range(0, len(items_list), chunk_size):
#                     yield items_list[i:i + chunk_size]
# #%%
# gen = shard_generator(dataset['train'])

# #%%
# for i in gen:
# #     print(i)
# #%%
# def r_none():
#     None

# for i in r_none():

#     print(i)