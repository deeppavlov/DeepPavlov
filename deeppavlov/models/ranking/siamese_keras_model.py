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
from keras import losses
from keras.optimizers import Adam
from keras.models import Model
from typing import List, Iterable
from abc import abstractmethod

from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.keras_model import KerasModel

log = get_logger(__name__)


class SiameseKerasModel(KerasModel):
    """The class implementing base functionality for siamese neural networks.

    Args:
        batch_size: A size of a batch.
        learning_rate: Learning rate.
        use_matrix: Whether to use trainable matrix with token (word) embeddings.
        emb_matrix: An embeddings matrix to initialize an embeddings layer of a model.
            Only used if ``use_matrix`` is set to ``True``.
        max_sequence_length: A maximum length of text sequences in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        dynamic_batch:  Whether to use dynamic batching. If ``True``, the maximum length of a sequence for a batch
            will be equal to the maximum of all sequences lengths from this batch,
            but not higher than ``max_sequence_length``.
        num_context_turns: A number of context turns in data samples.
        **kwargs: Other parameters.
    """

    def __init__(self,
                 batch_size: int,
                 learning_rate: float = 1e-3,
                 use_matrix: bool = True,
                 emb_matrix: np.ndarray = None,
                 max_sequence_length: int = None,
                 dynamic_batch: bool = False,
                 num_context_turns: int = 1,
                 **kwargs):

        self.save_path = kwargs.get('save_path', None)
        self.load_path = kwargs.get('load_path', None)
        train_now = kwargs.get('train_now', None)
        mode = kwargs.get('mode', None)

        super().__init__(save_path=self.save_path, load_path=self.load_path,
                         train_now=train_now, mode=mode)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_context_turns = num_context_turns
        self.train_now = train_now
        self.use_matrix = use_matrix
        self.emb_matrix = emb_matrix
        if dynamic_batch:
            self.max_sequence_length = None
        else:
            self.max_sequence_length = max_sequence_length
        self.model = self.create_model()
        self.compile()
        if self.load_path.exists():
           self.load()
        else:
            self.load_initial_emb_matrix()

    def compile(self) -> None:
        optimizer = Adam(lr=self.learning_rate)
        loss = losses.binary_crossentropy
        self.model.compile(loss=loss, optimizer=optimizer)

    def load(self) -> None:
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        self.model.load_weights(str(self.load_path))

    def save(self) -> None:
        log.info("[saving `{}`]".format(self.__class__.__name__))
        self.model.save_weights(str(self.save_path))

    def load_initial_emb_matrix(self) -> None:
        log.info("[initializing new `{}`]".format(self.__class__.__name__))
        if self.use_matrix:
            self.model.get_layer(name="embedding").set_weights([self.emb_matrix])

    @check_attr_true('train_now')
    def train_on_batch(self, batch: List[List[List[np.ndarray]]], y: List[int]) -> float:
        b = self.make_batch(batch)
        loss = self._train_on_batch(b, y)
        return loss

    def __call__(self, batch: Iterable[List[List[np.ndarray]]]) -> np.ndarray:
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
                        b = self.make_batch(buf[i*self.batch_size:(i+1)*self.batch_size])
                        yp = self._predict_on_batch(b)
                        y_pred += list(yp)
                    lenb = len(buf) % self.batch_size
                    if lenb != 0:
                        buf = buf[-lenb:]
                    else:
                        buf = []
            except StopIteration:
                if len(buf) != 0:
                    b = self.make_batch(buf)
                    yp = self._predict_on_batch(b)
                    y_pred += list(yp)
                break
        y_pred = np.asarray(y_pred)
        if len(responses) > 1:
            y_pred = np.reshape(y_pred, (j, len(responses)))
        return y_pred

    @abstractmethod
    def create_model(self) -> Model:
        pass

    def _train_on_batch(self, batch: List[np.ndarray], y: List[int]) -> float:
        loss = self.model.train_on_batch(batch, np.asarray(y))
        return loss

    def _predict_on_batch(self, batch: List[np.ndarray]) -> np.ndarray:
        y_pred = self.model.predict_on_batch(batch)
        return y_pred

    def reset(self) -> None:
        pass

    def make_batch(self, x: List[List[List[np.ndarray]]]) -> List[np.ndarray]:
        b = []
        for i in range(len(x[0])):
            z = [el[i] for el in x]
            b.append(np.asarray(z))
        return b