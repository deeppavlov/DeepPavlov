from pathlib import Path
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.models.ranking.ranking_dict import RankingDict
from collections import Counter
from nltk import word_tokenize
import csv
import re

class UbuntuV2Dict(RankingDict):

    def __init__(self, vocabs_path, save_path, load_path,
                 max_sequence_length, padding="post", truncating="post",
                 max_token_length=None, token_embeddings=True, char_embeddings=False,
                 char_pad="post", char_trunc="post",
                 tok_dynamic_batch=False, char_dynamic_batch=False,
                 tok_vocab_size=None):

        super().__init__(save_path, load_path,
                         max_sequence_length, padding, truncating,
                         max_token_length, token_embeddings, char_embeddings,
                         char_pad, char_trunc,
                         tok_dynamic_batch, char_dynamic_batch)

        vocabs_path = expand_path(vocabs_path)
        self.train_fname = Path(vocabs_path) / 'train.csv'
        self.val_fname = Path(vocabs_path) / 'valid.csv'
        self.test_fname = Path(vocabs_path) / 'test.csv'
        self.tok_vocab_size = tok_vocab_size
        self.int2char_vocab = dict()

    def build_int2char_vocab(self):
        sen = []
        with open(self.train_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                sen += el[:2]
        char_set = set()
        for el in sen:
            for x in el:
                char_set.add(x)
        self.int2char_vocab = {el[0]+1: el[1] for el in enumerate(char_set)}
        self.int2char_vocab[0] = '<UNK_CHAR>'

    def _find_most_common_words(self):
        sen = []
        with open(self.train_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                sen += el[:2]
        c = Counter()
        for el in sen:
            for x in el.split():
                c[x] += 1
        most_com_words = set([el[0] for el in c.most_common(self.tok_vocab_size-1)])
        return most_com_words

    def build_int2tok_vocab(self):
        if self.tok_vocab_size is not None:
            word_set = self._find_most_common_words()
        else:
            sen = []
            with open(self.train_fname, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for el in reader:
                    sen += el[:2]
            word_set = set()
            for el in sen:
                for x in el.split():
                    word_set.add(x)
        self.int2tok_vocab = {el[0]+1: el[1] for el in enumerate(word_set)}
        self.int2tok_vocab[0] = '<UNK>'

    def build_context2toks_vocabulary(self):
        self.context2toks_vocab = self._build_int2toks_vocabulary()

    def build_response2toks_vocabulary(self):
        self.response2toks_vocab = self._build_int2toks_vocabulary()

    def _build_int2toks_vocabulary(self):
        cont_train = []
        resp_train = []
        cont_valid = []
        resp_valid = []
        cont_test = []
        resp_test = []

        with open(self.train_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                cont_train.append(el[0])
                resp_train.append(el[1])
        with open(self.val_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                cont_valid.append(el[0])
                resp_valid += el[1:]
        with open(self.test_fname, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for el in reader:
                cont_test.append(el[0])
                resp_test += el[1:]

        sen = cont_train + resp_train + cont_valid + resp_valid + cont_test + resp_test
        int2toks_vocab = {el[0]: el[1].split() for el in enumerate(sen)}
        return int2toks_vocab
