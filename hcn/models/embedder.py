from gensim.models import word2vec
import numpy as np
from pathlib import Path

from deeppavlov.common import paths
from deeppavlov.common.registry import register_model
from deeppavlov.models.model import Model


@register_model('w2v')
class UtteranceEmbed(Model):
    def __init__(self, corpus_path, model_dir_path='emb', model_fpath='text8.model', dim=300,
                 train_now=False):
        self._corpus_path = corpus_path
        self._model_path = Path(paths.USR_PATH).joinpath(model_dir_path, model_fpath).as_posix()
        self.dim = dim
        self._train_now = train_now

        if self._train_now:
            self._create_model()
            self.model = word2vec.Word2Vec.load(self._model_path)

        else:
            try:
                self.model = word2vec.Word2Vec.load(self._model_path)
            except:
                print("There is no pretrained model, training a new one anyway.")
                self._create_model()
                self.model = word2vec.Word2Vec.load(self._model_path)

    def _encode(self, utterance):
        embs = [self.model[word] for word in utterance.split(' ') if word and word in self.model]
        # average of embeddings
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim], np.float32)

    def _create_model(self):
        sentences = word2vec.Text8Corpus(self._corpus_path)

        print(':: creating new word2vec model')
        model = word2vec.Word2Vec(sentences, size=self.dim)

        Path.mkdir(paths.USR_PATH.joinpath(self._model_dir_path))
        model.save(self._model_path)
        print(':: model saved to path')

    def infer(self, utterance):
        return self._encode(utterance)
