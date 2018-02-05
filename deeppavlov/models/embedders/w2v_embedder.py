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

import numpy as np
from gensim.models import word2vec

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.attributes import check_attr_true


@register('w2v')
class Word2VecEmbedder(Trainable, Inferable):
    def __init__(self, ser_path, ser_dir='emb', ser_file='text8.model', dim=300,
                 train_now=False):
        super().__init__(ser_path=ser_path, ser_dir=ser_dir,
                         ser_file=ser_file, train_now=train_now)
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
        sentences = word2vec.Text8Corpus(self.ser_path)

        print(':: creating new word2vec model')
        model = word2vec.Word2Vec(sentences, size=self.dim)
        self.model = model

        self.ser_path.parent.mkdir(parents=True, exist_ok=True)
        self.save()
        return model

    def infer(self, sentence: str, *args, **kwargs):
        return self._encode(sentence)

    # @run_alt_meth_if_no_path(train, 'train_now')
    def load(self):
        return word2vec.Word2Vec.load(str(self.ser_path))

    def save(self):
        self.model.save(str(self.ser_path))
        print(':: model saved to path')
