# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from pathlib import Path
from deeppavlov.core.models.inferable import Inferable


class DictEmbedder(Inferable):
    def __init__(self, embedding_dim, model_path=None, *args, **kwargs):
        """
        Method initializes the class according to given parameters.
        Args:
            embedding_dim: embedding dimension
        """
        self.tok2emb = {}
        self.embedding_dim = embedding_dim
        self.load_items(fname=model_path)

    def load_items(self, fname):
        """
        Method initializes dictionary of embeddings from file.
        Returns:
            Nothing
        """

        if fname is None or not Path(fname).is_file():
            raise IOError('There is no dictionary of embeddings <<%s>> file provided.' % fname)
        else:
            print('Loading existing dictionary of embeddings from %s.' % fname)
            with open(fname, 'r') as f:
                for line in f:
                    values = line.rsplit(sep=' ', maxsplit=self.embedding_dim)
                    assert(len(values) == self.embedding_dim + 1)
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.tok2emb[word] = coefs
