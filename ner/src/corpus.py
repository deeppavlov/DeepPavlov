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

from collections import Counter
from collections import defaultdict
import random
import numpy as np


DATA_PATH = '/tmp/nerpa'
DOC_START_STRING = '-DOCSTART-'
SEED = 42
SPECIAL_TOKENS = ['<PAD>', '<UNK>']
SPECIAL_TAGS = ['<PAD>']

np.random.seed(SEED)
random.seed(SEED)


# Dictionary class. Each instance holds tags or tokens or characters and provides
# dictionary like functionality like indices to tokens and tokens to indices.
class Vocabulary:
    def __init__(self, tokens=None, default_token='<UNK>', is_tags=False):
        if is_tags:
            special_tokens = SPECIAL_TAGS
            self._t2i = dict()
        else:
            special_tokens = SPECIAL_TOKENS
            if default_token not in special_tokens:
                raise Exception('SPECIAL_TOKENS must contain <UNK> token!')
            # We set default ind to position of <UNK> in SPECIAL_TOKENS
            # because the tokens will be added to dict in the same order as
            # in SPECIAL_TOKENS
            default_ind = special_tokens.index('<UNK>')
            self._t2i = defaultdict(lambda: default_ind)
        self._i2t = dict()
        self.frequencies = Counter()

        self.counter = 0
        for token in special_tokens:
            self._t2i[token] = self.counter
            self.frequencies[token] += 0
            self._i2t[self.counter] = token
            self.counter += 1
        if tokens is not None:
            self.update_dict(tokens)

    def update_dict(self, tokens):
        for token in tokens:
            if token not in self._t2i:
                self._t2i[token] = self.counter
                self._i2t[self.counter] = token
                self.counter += 1
            self.frequencies[token] += 1

    def idx2tok(self, idx):
        return self._i2t[idx]

    def idxs2toks(self, idxs, filter_paddings=False):
        toks = []
        for idx in idxs:
            if not filter_paddings or idx != self.tok2idx('<PAD>'):
                toks.append(self._i2t[idx])
        return toks

    def tok2idx(self, tok):
        return self._t2i[tok]

    def toks2idxs(self, toks):
        return [self._t2i[tok] for tok in toks]

    def batch_toks2batch_idxs(self, b_toks):
        max_len = max(len(toks) for toks in b_toks)
        # Create array filled with paddings
        batch = np.ones([len(b_toks), max_len]) * self.tok2idx('<PAD>')
        for n, tokens in enumerate(b_toks):
            idxs = self.toks2idxs(tokens)
            batch[n, :len(idxs)] = idxs
        return batch

    def batch_idxs2batch_toks(self, b_idxs, filter_paddings=False):
        return [self.idxs2toks(idxs, filter_paddings) for idxs in b_idxs]

    def is_pad(self, x_t):
        assert type(x_t) == np.ndarray
        return x_t == self.tok2idx('<PAD>')

    def __getitem__(self, key):
        return self._t2i[key]

    def __len__(self):
        return self.counter

    def __contains__(self, item):
        return item in self._t2i


class Corpus:
    def __init__(self, dataset=None, embeddings_file_path=None, dicts_filepath=None):
        if dataset is not None:
            self.dataset = dataset
            self.token_dict = Vocabulary(self.get_tokens())
            self.tag_dict = Vocabulary(self.get_tags(), is_tags=True)
            self.char_dict = Vocabulary(self.get_characters())
        elif dicts_filepath is not None:
            self.dataset = None
            self.load_corpus_dicts(dicts_filepath)

        if embeddings_file_path is not None:
            self.embeddings = self.load_embeddings(embeddings_file_path)
        else:
            self.embeddings = None

    # All tokens for dictionary building
    def get_tokens(self, data_type='train'):
        for tokens, _ in self.dataset[data_type]:
            for token in tokens:
                yield token

    # All tags for dictionary building
    def get_tags(self, data_type=None):
        if data_type is None:
            data_types = self.dataset.keys()
        else:
            data_types = [data_type]
        for data_type in data_types:
            for _, tags in self.dataset[data_type]:
                for tag in tags:
                    yield tag

    # All characters for dictionary building
    def get_characters(self, data_type='train'):
        for tokens, _ in self.dataset[data_type]:
            for token in tokens:
                for character in token:
                    yield character

    def load_embeddings(self, file_path):
        # Embeddins must be in fastText format either bin or
        print('Loading embeddins...')
        if file_path.endswith('.bin'):
            from gensim.models.wrappers import FastText
            embeddings = FastText.load_fasttext_format(file_path)
        else:
            pre_trained_embeddins_dict = dict()
            with open(file_path) as f:
                _ = f.readline()
                for line in f:
                    token, *embedding = line.split()
                    embedding = np.array([float(val_str) for val_str in embedding])
                    if token in self.token_dict:
                        pre_trained_embeddins_dict[token] = embedding
            print('Readed')
            pre_trained_std = np.std(list(pre_trained_embeddins_dict.values()))
            embeddings = pre_trained_std * np.random.randn(len(self.token_dict), len(embedding))
            for idx in range(len(self.token_dict)):
                token = self.token_dict.idx2tok(idx)
                if token in pre_trained_embeddins_dict:
                    embeddings[idx] = pre_trained_embeddins_dict[token]
        return embeddings

    def tokens_to_x_and_xc(self, tokens):
        n_tokens = len(tokens)
        tok_idxs = self.token_dict.toks2idxs(tokens)
        char_idxs = []
        max_char_len = 0
        for token in tokens:
            char_idxs.append(self.char_dict.toks2idxs(token))
            max_char_len = max(max_char_len, len(token))
        toks = np.zeros([1, n_tokens], dtype=np.int32)
        chars = np.zeros([1, n_tokens, max_char_len], dtype=np.int32)
        toks[0, :] = tok_idxs
        for n, char_line in enumerate(char_idxs):
            chars[0, n, :len(char_line)] = char_line
        return toks, chars

    def batch_generator(self,
                        batch_size,
                        dataset_type='train',
                        shuffle=True,
                        allow_smaller_last_batch=True,
                        return_char=True):
        tokens_tags_pairs = self.dataset[dataset_type]
        n_samples = len(tokens_tags_pairs)
        if shuffle:
            order = np.random.permutation(n_samples)
        else:
            order = np.arange(n_samples)
        n_batches = n_samples // batch_size
        if allow_smaller_last_batch and n_samples % batch_size:
            n_batches += 1
        for k in range(n_batches):
            batch_start = k * batch_size
            batch_end = min((k + 1) * batch_size, n_samples)
            x_batch = [tokens_tags_pairs[ind][0] for ind in order[batch_start: batch_end]]
            y_batch = [tokens_tags_pairs[ind][1] for ind in order[batch_start: batch_end]]
            (x_token, x_char), y = self.tokens_batch_to_numpy_batch(x_batch, y_batch, return_char)
            if return_char:
                yield (x_token, x_char), y
            else:
                yield x_token, y

    def tokens_batch_to_numpy_batch(self, batch_x, batch_y=None, return_char=True):
        # Determine dimensions
        batch_size = len(batch_x)
        max_utt_len = max([len(utt) for utt in batch_x])
        max_token_len = max([len(token) for utt in batch_x for token in utt])
        prepare_embeddings_onthego = self.embeddings is not None and not isinstance(self.embeddings, dict)
        # Prepare numpy arrays
        if prepare_embeddings_onthego:  # If the embeddings is a fastText model
            x_token = np.zeros([batch_size, max_utt_len, self.embeddings.vector_size], dtype=np.float32)
        else:  # If the embeddings is a token - vector dictionary
            x_token = np.ones([batch_size, max_utt_len], dtype=np.int32) * self.token_dict['<PAD>']
        if return_char:
            x_char = np.ones([batch_size, max_utt_len, max_token_len], dtype=np.int32) * self.char_dict['<PAD>']
        else:
            x_char = None
        if batch_y is not None:
            y = np.ones([batch_size, max_utt_len], dtype=np.int32) * self.tag_dict['<PAD>']
        else:
            y = None

        # Prepare x batch
        for n, utterance in enumerate(batch_x):
            if prepare_embeddings_onthego:
                try:
                    x_token[n, :len(utterance), :] = [self.embeddings[token] for token in utterance]
                except KeyError:
                    pass
            else:
                x_token[n, :len(utterance)] = self.token_dict.toks2idxs(utterance)
            if return_char:
                for k, token in enumerate(utterance):
                    x_char[n, k, :len(token)] = self.char_dict.toks2idxs(token)

        # Prepare y batch
        if batch_y is not None:
            for n, tags in enumerate(batch_y):
                y[n, :len(tags)] = self.tag_dict.toks2idxs(tags)

        return (x_token, x_char), y

    def save_corpus_dicts(self, filename='dict.txt'):
        # Token dict
        token_dict = self.token_dict._i2t
        with open(filename, 'w') as f:
            f.write('-TOKEN-DICT-\n')
            for ind in range(len(token_dict)):
                f.write(token_dict[ind] + '\n')
            f.write('\n')

        # Tag dict
        token_dict = self.tag_dict._i2t
        with open(filename, 'a') as f:
            f.write('-TAG-DICT-\n')
            for ind in range(len(token_dict)):
                f.write(token_dict[ind] + '\n')
            f.write('\n')

        # Character dict
        token_dict = self.char_dict._i2t
        with open(filename, 'a') as f:
            f.write('-CHAR-DICT-\n')
            for ind in range(len(token_dict)):
                f.write(token_dict[ind] + '\n')
            f.write('\n')

    def load_corpus_dicts(self, filename='dict.txt'):
        with open(filename) as f:
            # Token dict
            tokens = list()
            line = f.readline()
            assert line.strip() == '-TOKEN-DICT-'
            while len(line) > 0:
                line = f.readline().strip()
                if len(line) > 0:
                    tokens.append(line)
            self.token_dict = Vocabulary(tokens)

            # Tag dictappend
            line = f.readline()
            tags = list()
            assert line.strip() == '-TAG-DICT-'
            while len(line) > 0:
                line = f.readline().strip()
                if len(line) > 0:
                    tags.append(line)
            self.tag_dict = Vocabulary(tags, is_tags=True)

            # Char dict
            line = f.readline()
            chars = list()
            assert line.strip() == '-CHAR-DICT-'
            while len(line) > 0:
                line = f.readline().strip()
                if len(line) > 0:
                    chars.append(line)
            self.char_dict = Vocabulary(chars)
