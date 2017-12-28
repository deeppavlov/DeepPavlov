from collections import Counter, defaultdict
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.attributes import check_path_exists


@register('default_vocab')
class DefaultVocabulary(Trainable, Inferable):

    def __init__(self, special_tokens=('UNK',), default_token='<UNK>', model_dir='',
                 model_file='vocab.txt'):
        self._model_dir = model_dir
        self._model_file = model_file
        self.special_tokens = special_tokens
        self.default_token = default_token
        
        self._reset_dict()
	
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._i2t[key]
        elif isinstance(key, str):
            return self._t2i[key]
        else:
            return NotImplemented("not implemented for type `{}`".format(type(key)))

    def __contains__(self, item):
        return item in self._t2i

    def __len__(self):
        return len(self.freqs)

    def _reset_dict(self):
        def constant_factory(value):
            return itertools.repeat(value).next

	# default index is the position of default_token
        default_ind = self.special_tokens.index(default_token)
        self._t2i = defaultdict(constant_factory(default_ind))
        self._i2t = dict()
        self.freqs = Counter()

        for i, token in enumerate(self.special_tokens):
            self._t2i[token] = i
            self._i2t[self.counter] = token
            self.freqs[token] = 0

    @overrides
    def train(self, tokens, counts=None, update=True):
        counts = counts or [1]*len(tokens)
        if not update:
            self._reset_dict()

        index = len(self.freqs)
        for t, cnt in zip(tokens, counts):
            if token not in self._t2i:
                self._t2i[token] = index
                self._i2t[index] = token
                index += 1
                self.freqs[token] += cnt

    @overrides
    def infer(self, samples):
        return [self.__getitem__(s) for s in samples]

    @overrides
    def save(self):
        with open(self._model_path, 'wt') as f:
            for token, cnt in self.freqs.most_common():
                f.write('{}\t{:d}\n'.format(token, cnt))

    @check_path_exists
    @overrides
    def load(self):
        tokens, counts = [], []
        for ln in open(self._model_path, 'r'):
            token, cnt = ln.split('\t', 1)
            tokens.append(token)
            counts.append(int(cnt))
        self.train(tokens=tokens, counts=counts, update=True)
