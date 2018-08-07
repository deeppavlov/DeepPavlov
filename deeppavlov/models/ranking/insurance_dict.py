from pathlib import Path
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.models.ranking.ranking_dict import RankingDict


class InsuranceDict(RankingDict):

    def __init__(self, vocabs_path, save_path, load_path,
                 max_sequence_length, padding="post", truncating="post",
                 max_token_length=None, token_embeddings=True, char_embeddings=False,
                 char_pad="post", char_trunc="post",
                 tok_dynamic_batch=False, char_dynamic_batch=False, update_embeddings = False):

        super().__init__(save_path, load_path,
                         max_sequence_length, max_token_length,
                         padding, truncating,
                         token_embeddings, char_embeddings,
                         char_pad, char_trunc,
                         tok_dynamic_batch, char_dynamic_batch, update_embeddings)

        vocabs_path = expand_path(vocabs_path)
        self.int2tok_fname = Path(vocabs_path) / 'vocabulary'
        self.response2ints_fname = Path(vocabs_path) / 'answers.label.token_idx'
        self.train_context2ints_fname = Path(vocabs_path) / 'question.train.token_idx.label'
        self.val_context2ints_fname = Path(vocabs_path) / 'question.dev.label.token_idx.pool'
        self.test_context2ints_fname = Path(vocabs_path) / 'question.test1.label.token_idx.pool'


    def build_int2tok_vocab(self):
        with open(self.int2tok_fname, 'r') as f:
            data = f.readlines()
        self.int2tok_vocab = {int(el.split('\t')[0].split('_')[1]): el.split('\t')[1][:-1] for el in data}
        self.int2tok_vocab[0] = '<UNK>'

    def build_response2toks_vocabulary(self):
        with open(self.response2ints_fname, 'r') as f:
            data = f.readlines()
            response2idxs_vocab = {int(el.split('\t')[0]) - 1:
                                   (el.split('\t')[1][:-1]).split(' ') for el in data}
        self.response2toks_vocab = {el[0]: [self.int2tok_vocab[int(x.split('_')[1])]
                                    for x in el[1]] for el in response2idxs_vocab.items()}

    def build_context2toks_vocabulary(self):
        contexts = []
        with open(self.train_context2ints_fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            c, _ = eli.split('\t')
            contexts.append(c.split(' '))
        with open(self.val_context2ints_fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            _, c, _ = eli.split('\t')
            contexts.append(c.split(' '))
        with open(self.test_context2ints_fname, 'r') as f:
            data = f.readlines()
        for eli in data:
            eli = eli[:-1]
            _, c, _ = eli.split('\t')
            contexts.append(c.split(' '))
        self.context2toks_vocab = {el[0]: [self.int2tok_vocab[int(x.split('_')[1])]
                                   for x in el[1]] for el in enumerate(contexts)}

    def build_int2char_vocab(self):
        pass