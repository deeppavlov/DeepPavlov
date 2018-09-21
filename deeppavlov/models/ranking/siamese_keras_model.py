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
from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.log import get_logger
from typing import Callable
from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.models.ranking.utils import make_batch
from keras import losses
from keras.optimizers import Adam

log = get_logger(__name__)


class SiameseKerasModel(KerasModel):
    """Class to perform ranking.

    Args:
        interact_pred_num: The number of the most relevant contexts and responses
            which model returns in the `interact` regime.
        update_embeddings: Whether to store and update context and response embeddings or not.

        **kwargs: Other parameters.
    """

    def __init__(self,
                 batch_size: int,
                 learning_rate: float = 1e-3,
                 use_matrix: bool = True,
                 emb_matrix: np.ndarray = None,
                 num_context_turns: int = 1,
                 preprocess: Callable = None,
                 context2emb_vocab: dict = None,
                 response2emb_vocab: dict = None,
                 update_embeddings: bool = False,
                 interact_pred_num: int = 3,
                 **kwargs):

        # Parameters for parent classes
        self.save_path = kwargs.get('save_path', None)
        self.load_path = kwargs.get('load_path', None)
        train_now = kwargs.get('train_now', None)
        mode = kwargs.get('mode', None)

        super().__init__(save_path=self.save_path, load_path=self.load_path,
                         train_now=train_now, mode=mode)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_context_turns = num_context_turns
        self.preprocess = preprocess
        self.interact_pred_num = interact_pred_num
        self.train_now = train_now
        self.update_embeddings = update_embeddings
        self.context2emb_vocab = context2emb_vocab
        self.response2emb_vocab = response2emb_vocab
        self.use_matrix = use_matrix
        self.emb_matrix = emb_matrix
        self.model = self.create_model()
        self.compile()
        if self.load_path.exists():
           self.load()
        else:
            self.load_initial_emb_matrix()

    def compile(self):
        optimizer = Adam(lr=self.learning_rate)
        loss = losses.binary_crossentropy
        self.model.compile(loss=loss, optimizer=optimizer)

    def load(self):
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        self.model.load_weights(str(self.load_path))

    def save(self):
        log.info("[saving `{}`]".format(self.__class__.__name__))
        self.model.save_weights(str(self.save_path))

    def load_initial_emb_matrix(self):
        log.info("[initializing new `{}`]".format(self.__class__.__name__))
        if self.use_matrix:
            self.model.get_layer(name="embedding").set_weights([self.emb_matrix])

    @check_attr_true('train_now')
    def train_on_batch(self, batch, y):
        """Train the model on a batch."""
        b = make_batch(batch)
        loss = self._train_on_batch(b, y)
        return loss

    def __call__(self, batch):
        """Make a prediction on a batch."""
        y_pred = []
        buf = []
        j = 0
        while True:
            try:
                el = next(batch)
                j += 1
                context = el[:self.num_context_turns]
                responses = el[self.num_context_turns:]
                buf += [context + [el] for el in responses]
                if len(buf) >= self.batch_size:
                    for i in range(len(buf) // self.batch_size):
                        b = make_batch(buf[i*self.batch_size:(i+1)*self.batch_size])
                        yp = self._predict_on_batch(b)
                        y_pred += list(yp)
                    lenb = len(buf) % self.batch_size
                    if lenb != 0:
                        buf = buf[-lenb:]
                    else:
                        buf = []
            except StopIteration:
                if len(buf) != 0:
                    b = make_batch(buf)
                    yp = self._predict_on_batch(b)
                    y_pred += list(yp)
                break
        y_pred = np.asarray(y_pred)
        if len(responses) > 1:
            y_pred = np.reshape(y_pred, (j, len(responses)))
        return y_pred

    def create_model(self):
        pass

    def _train_on_batch(self, batch, y):
        self.model.train_on_batch(batch, np.asarray(y))

    def _predict_on_batch(self, batch):
        loss = self.model.predict_on_batch(batch)
        return loss

    def reset(self):
        pass
