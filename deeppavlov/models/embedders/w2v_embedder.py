from pathlib import Path

import numpy as np
from gensim.models import word2vec

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.attributes import check_path_exists, check_attr_true,\
    run_alt_meth_if_no_path


@register('w2v')
class UtteranceEmbed(Trainable, Inferable):
    def __init__(self, corpus_path, model_dir='emb', model_file='text8.model', dim=300):
        self._corpus_path = corpus_path
        self._model_dir = model_dir
        self._model_file = model_file
        self.dim = dim
        self.model = self.load()

    def _encode(self, utterance):
        embs = [self.model[word] for word in utterance.split(' ') if word and word in self.model]
        # average of embeddings
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim], np.float32)

    @check_attr_true('train_now')
    def train(self):
        sentences = word2vec.Text8Corpus(self._corpus_path)

        print(':: creating new word2vec model')
        model = word2vec.Word2Vec(sentences, size=self.dim)
        self.model = model

        if not self.model_path.parent.exists():
            Path.mkdir(self.model_path.parent)

        self.save()

    def infer(self, utterance):
        return self._encode(utterance)

    @run_alt_meth_if_no_path(train, 'train_now')
    def load(self):
        return word2vec.Word2Vec.load(str(self.model_path))

    def save(self):
        self.model.save(str(self.model_path))
        print(':: model saved to path')
