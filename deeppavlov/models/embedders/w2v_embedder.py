import numpy as np
from gensim.models import word2vec

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.attributes import check_attr_true, run_alt_meth_if_no_path


@register('w2v')
class Word2VecEmbedder(Trainable, Inferable):
    def __init__(self, corpus_path, model_dir='emb', model_file='text8.model', dim=300):
        self._corpus_path = corpus_path
        self._model_dir = model_dir
        self._model_file = model_file
        self.dim = dim
        self.model = self.load()

    def _encode(self, sentence: str):
        embs = [self.model[w] for w in sentence.split() if w and w in self.model]
        # average of embeddings
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim], np.float32)

    @check_attr_true('train_now')
    def train(self, *args, **kwargs):
        sentences = word2vec.Text8Corpus(self._corpus_path)

        print(':: creating new word2vec model')
        model = word2vec.Word2Vec(sentences, size=self.dim)
        self.model = model

        self.make_dir()
        self.save()
        return model

    def infer(self, sentence: str, *args, **kwargs):
        return self._encode(sentence)

    @run_alt_meth_if_no_path(train, 'train_now')
    def load(self):
        return word2vec.Word2Vec.load(str(self.model_path_))

    def save(self):
        self.model.save(str(self.model_path_))
        print(':: model saved to path')
