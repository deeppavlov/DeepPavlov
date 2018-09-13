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

from overrides import overrides
from copy import deepcopy
import inspect
import numpy as np

from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.ranking.siamese_network import SiameseNetwork
from deeppavlov.core.common.log import get_logger
from typing import Union, List, Tuple, Dict, Callable

log = get_logger(__name__)


@register('siamese_model')
class SiameseModel(NNModel):
    """Class to perform ranking.

    Args:
        interact_pred_num: The number of the most relevant contexts and responses
            which model returns in the `interact` regime.
        update_embeddings: Whether to store and update context and response embeddings or not.

        **kwargs: Other parameters.
    """

    def __init__(self,
                 num_context_turns: int = 1,
                 preprocess: Callable = None,
                 context2emb_vocab: dict = None,
                 response2emb_vocab: dict = None,
                 update_embeddings: bool = False,
                 interact_pred_num: int = 3,
                 **kwargs):

        # Parameters for parent classes
        save_path = kwargs.get('save_path', None)
        load_path = kwargs.get('load_path', None)
        train_now = kwargs.get('train_now', None)
        mode = kwargs.get('mode', None)

        super().__init__(save_path=save_path, load_path=load_path,
                         train_now=train_now, mode=mode)

        self.num_context_turns = num_context_turns
        self.preprocess = preprocess
        self.interact_pred_num = interact_pred_num
        self.train_now = train_now
        self.update_embeddings = update_embeddings
        self.context2emb_vocab = context2emb_vocab
        self.response2emb_vocab = response2emb_vocab

        # opt = deepcopy(kwargs)

        network_parameter_names = list(inspect.signature(SiameseNetwork.__init__).parameters)
        self.network_parameters = {par: kwargs[par] for par in network_parameter_names if par in kwargs}

        self.load()

        train_parameters_names = list(inspect.signature(self._net.train_on_batch).parameters)
        self.train_parameters = {par: kwargs[par] for par in train_parameters_names if par in kwargs}

    def load(self):
        """Load the model from the last checkpoint if it exists. Otherwise instantiate a new model."""
        self._net = SiameseNetwork(num_context_turns=self.num_context_turns, **self.network_parameters)

    def save(self):
        """Save the model."""
        log.info('[saving model to {}]'.format(self.save_path.resolve()))
        self._net.save()
        if self.update_embeddings:
            self.update_sen_embs(self.context2emb_vocab, "context")
            self.update_sen_embs(self.response2emb_vocab, "response")
        # self.embdict.save()

    @check_attr_true('train_now')
    def train_on_batch(self, batch, y):
        """Train the model on a batch."""
        b = self.make_batch(batch)
        loss = self._net.train_on_batch(b, y)
        return loss

    def __call__(self, batch):
        """Make a prediction on a batch."""
        if len(batch) > 1:
            y_pred = []
            b = self.make_batch(batch)
            for el in b[self.num_context_turns:]:
                yp = self._net.predict_score_on_batch(b[:self.num_context_turns] + [el])
                if len(b[self.num_context_turns:]) > 1:
                    yp = np.expand_dims(yp, 1)
                y_pred.append(yp)
            y_pred = np.hstack(y_pred)
            return y_pred

        # else:
        #     c_input = tokenize(batch)
        #     c_input = self.dict.make_ints(c_input)
        #     c_input_emb = self._net.predict_embedding_on_batch([c_input, c_input], type='context')
        #
        #     c_emb = [self.dict.context2emb_vocab[i] for i in range(len(self.dict.context2emb_vocab))]
        #     c_emb = np.vstack(c_emb)
        #     pred_cont = np.sum(c_input_emb * c_emb, axis=1)\
        #              / np.linalg.norm(c_input_emb, axis=1) / np.linalg.norm(c_emb, axis=1)
        #     pred_cont = np.flip(np.argsort(pred_cont), 0)[:self.interact_pred_num]
        #     pred_cont = [' '.join(self.dict.context2toks_vocab[el]) for el in pred_cont]
        #
        #     r_emb = [self.dict.response2emb_vocab[i] for i in range(len(self.dict.response2emb_vocab))]
        #     r_emb = np.vstack(r_emb)
        #     pred_resp = np.sum(c_input_emb * r_emb, axis=1)\
        #              / np.linalg.norm(c_input_emb, axis=1) / np.linalg.norm(r_emb, axis=1)
        #     pred_resp = np.flip(np.argsort(pred_resp), 0)[:self.interact_pred_num]
        #     pred_resp = [' '.join(self.dict.response2toks_vocab[el]) for el in pred_resp]
        #     y_pred = [{"contexts": pred_cont, "responses": pred_resp}]
        #     return y_pred

    def update_sen_embs(self, sen2emb_vocab, type):
        bs = 512
        r = list(sen2emb_vocab.keys())
        num_batches = len(r) // bs
        sen_embeddings = []
        for i in range(num_batches):
            sen = r[i * bs: (i+1) * bs]
            batch = self.preprocess(list(zip([sen, sen])))
            sen_embeddings.append(self._net.predict_embedding_on_batch(batch, type=type))
        if len(r) % bs != 0:
            sen = r[num_batches * bs:]
            batch = self.preprocess(zip([sen, sen]))
            sen_embeddings.append(self._net.predict_embedding_on_batch(batch, type=type))
        sen_embeddings = np.vstack(sen_embeddings)
        for i, el in enumerate(r):
            sen2emb_vocab[el] = sen_embeddings[i]

    def shutdown(self):
        pass

    def reset(self):
        pass

    def make_batch(self, x):
        b = []
        for i in range(len(x[0])):
            z = [el[i] for el in x]
            b.append(np.asarray(z))
        return b