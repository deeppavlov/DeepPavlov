from gensim.models import word2vec
import numpy as np

from deeppavlov.common.registry import register_model
from deeppavlov.models.model import Model

from typing import Dict
from inspect import getfullargspec

from hcn.paths import TEXT8_MODEL


@register_model('w2v')
class UtteranceEmbed(Model):
    def __init__(self,
                 fname=TEXT8_MODEL,
                 dim=300):
        self.dim = dim
        try:
            # load saved model
            self.model = word2vec.Word2Vec.load(fname)
        except:
            print(':: creating new word2vec model')
            self.create_model()
            self.model = word2vec.Word2Vec.load(fname)

    def _encode(self, utterance):
        embs = [self.model[word] for word in utterance.split(' ') if word and word in self.model]
        # average of embeddings
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim], np.float32)

    # def create_model(self, fname='text8'):
    #     sentences = word2vec.Text8Corpus('path')
    #     model = word2vec.Word2Vec(sentences, size=self.dim)
    #     model.save('path')
    #     print(':: model saved to path')

    def infer(self, utterance):
        return self._encode(utterance)

