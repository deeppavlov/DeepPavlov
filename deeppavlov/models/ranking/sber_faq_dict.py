from pathlib import Path
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.models.ranking.ranking_dict import RankingDict
from nltk import word_tokenize
import csv
import re

class SberFAQDict(RankingDict):

    def __init__(self, vocabs_path, save_path, load_path,
                 max_sequence_length, padding="post", truncating="post",
                 max_token_length=None, token_embeddings=True, char_embeddings=False,
                 char_pad="post", char_trunc="post",
                 tok_dynamic_batch=False, char_dynamic_batch=False):

        super().__init__(save_path, load_path,
                         max_sequence_length, padding, truncating,
                         max_token_length, token_embeddings,
                         char_pad, char_trunc,
                         tok_dynamic_batch, char_dynamic_batch)

        vocabs_path = expand_path(vocabs_path)
        self.train_fname = Path(vocabs_path) / 'sber_faq_train_1849.csv'
        self.val_fname = Path(vocabs_path) / 'sber_faq_val_1849.csv'
        self.test_fname = Path(vocabs_path) / 'sber_faq_test_1849.csv'

    def build_int2char_vocab(self):
        sen = []
        with open(self.train_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(self.clean_sen(el[0]))
        with open(self.val_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(self.clean_sen(el[0]))
        char_set = set()
        for el in sen:
            for x in el:
                char_set.add(x)
        self.int2char_vocab = {el[0]+1: el[1] for el in enumerate(char_set)}
        self.int2char_vocab[0] = '<UNK_CHAR>'


    def build_int2tok_vocab(self):
        sen = []
        with open(self.train_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(self.clean_sen(el[0]))
        with open(self.val_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(self.clean_sen(el[0]))
        word_set = set()
        for el in sen:
            for x in word_tokenize(el):
                word_set.add(x)
        self.int2tok_vocab = {el[0]+1: el[1] for el in enumerate(word_set)}
        self.int2tok_vocab[0] = '<UNK>'

    def build_context2toks_vocabulary(self):
        self.context2toks_vocab = self._build_int2toks_vocabulary()

    def build_response2toks_vocabulary(self):
        self.response2toks_vocab = self._build_int2toks_vocabulary()

    def _build_int2toks_vocabulary(self):
        sen = []
        with open(self.train_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(self.clean_sen(el[0]))
        with open(self.val_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(self.clean_sen(el[0]))
        with open(self.test_fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for el in reader:
                sen.append(self.clean_sen(el[0]))
        int2toks_vocab = {el[0]: word_tokenize(el[1]) for el in enumerate(sen)}
        return int2toks_vocab

    def clean_sen(self, sen):
        return re.sub('\[Клиент:.*\]', '', sen).replace('&amp, laquo, ', '').replace('&amp, laquo, ', '').\
            replace('&amp laquo ', '').replace('&amp quot ', '').replace('&amp quot ', '').strip()
