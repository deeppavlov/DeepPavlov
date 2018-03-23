import pickle
from pathlib import Path
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.models.ranking.ranking_dict import RankingDict

class UbuntuDict(RankingDict):

    def __init__(self, vocabs_path, save_path, load_path,
                 max_sequence_length, padding="post", truncating="pre"):

        super().__init__(save_path, load_path,
              max_sequence_length, padding, truncating)

        vocabs_path = expand_path(vocabs_path)
        self.int2tok_fname = Path(vocabs_path) / 'W.pkl'
        self.response2ints_fname = Path(vocabs_path) / 'dataset.pkl'
        self.context2ints_fname = Path(vocabs_path) / 'dataset.pkl'

    def build_int2tok_vocab(self):
        with open(self.int2tok_fname, 'rb') as f:
            W_data = pickle.load(f, encoding='bytes')
        bwords = [el[0] for el in W_data[1].items()]
        toks = ['<UNK>'] + [el.decode("utf-8") for el in bwords]
        self.int2tok_vocab = {el[0]: el[1] for el in enumerate(toks)}

    def build_response2toks_vocabulary(self):
        with open(self.response2ints_fname, 'rb') as f:
            data = pickle.load(f)
        a = list(zip(data[0]['c'], data[0]['r'], data[0]['y']))
        a = list(filter(lambda x: len(x[1]) != 0, a))
        data[0]['c'], data[0]['r'], data[0]['y'] = zip(*a)
        data[0]['r'] = list(data[0]['r'])
        all_resps = data[0]['r'] + data[1]['r'] + data[2]['r']
        all_resps = sorted(set([' '.join(map(str, el)) for el in all_resps]))
        vocab = {el[0]: el[1] for el in enumerate(all_resps)}
        self.response2toks_vocab = {el[0]: [self.int2tok_vocab[int(x)]
                                    for x in el[1].split(' ')] for el in vocab.items()}

    def build_context2toks_vocabulary(self):
        with open(self.context2ints_fname, 'rb') as f:
            data = pickle.load(f)
        a = list(zip(data[0]['c'], data[0]['r'], data[0]['y']))
        a = list(filter(lambda x: len(x[1]) != 0, a))
        data[0]['c'], data[0]['r'], data[0]['y'] = zip(*a)
        data[0]['c'] = list(data[0]['c'])
        all_conts = data[0]['c'] + data[1]['c'] + data[2]['c']
        all_conts = sorted(set([' '.join(map(str, el)) for el in all_conts]))
        vocab = {el[0]: el[1] for el in enumerate(all_conts)}
        self.context2toks_vocab = {el[0]: [self.int2tok_vocab[int(x)]
                                    for x in el[1].split(' ')] for el in vocab.items()}
