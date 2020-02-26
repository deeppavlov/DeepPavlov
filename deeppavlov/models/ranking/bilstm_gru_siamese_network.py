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

from logging import getLogger

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, GlobalMaxPooling1D, Lambda, Dense, GRU
from tensorflow.keras.models import Model

from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.bilstm_siamese_network import BiLSTMSiameseNetwork

log = getLogger(__name__)


@register('bilstm_gru_nn')
class BiLSTMGRUSiameseNetwork(BiLSTMSiameseNetwork):
    """The class implementing a siamese neural network with BiLSTM, GRU and max pooling.

    GRU is used to take into account multi-turn dialogue ``context``.

    Args:
        len_vocab: A size of the vocabulary to build embedding layer.
        seed: Random seed.
        shared_weights: Whether to use shared weights in the model to encode ``contexts`` and ``responses``.
        embedding_dim: Dimensionality of token (word) embeddings.
        reccurent: A type of the RNN cell. Possible values are ``lstm`` and ``bilstm``.
        hidden_dim: Dimensionality of the hidden state of the RNN cell. If ``reccurent`` equals ``bilstm``
            ``hidden_dim`` should be doubled to get the actual dimensionality.
        max_pooling: Whether to use max-pooling operation to get ``context`` (``response``) vector representation.
            If ``False``, the last hidden state of the RNN will be used.
        triplet_loss: Whether to use a model with triplet loss.
            If ``False``, a model with crossentropy loss will be used.
        margin: A margin parameter for triplet loss. Only required if ``triplet_loss`` is set to ``True``.
        hard_triplets: Whether to use hard triplets sampling to train the model
            i.e. to choose negative samples close to positive ones.
            If set to ``False`` random sampling will be used.
            Only required if ``triplet_loss`` is set to ``True``.
    """

    def create_model(self) -> Model:
        input = []
        if self.use_matrix:
            for i in range(self.num_context_turns + 1):
                input.append(Input(shape=(self.max_sequence_length,)))
            context = input[:self.num_context_turns]
            response = input[-1]
            emb_layer = self.embedding_layer()
            emb_c = [emb_layer(el) for el in context]
            emb_r = emb_layer(response)
        else:
            for i in range(self.num_context_turns + 1):
                input.append(Input(shape=(self.max_sequence_length, self.embedding_dim,)))
            context = input[:self.num_context_turns]
            response = input[-1]
            emb_c = context
            emb_r = response
        lstm_layer = self.lstm_layer()
        lstm_c = [lstm_layer(el) for el in emb_c]
        lstm_r = lstm_layer(emb_r)
        pooling_layer = GlobalMaxPooling1D(name="pooling")
        lstm_c = [pooling_layer(el) for el in lstm_c]
        lstm_r = pooling_layer(lstm_r)
        lstm_c = [Lambda(lambda x: K.expand_dims(x, 1))(el) for el in lstm_c]
        lstm_c = Lambda(lambda x: K.concatenate(x, 1))(lstm_c)
        gru_layer = GRU(2 * self.hidden_dim, name="gru")
        gru_c = gru_layer(lstm_c)

        if self.triplet_mode:
            dist = Lambda(self._pairwise_distances)([gru_c, lstm_r])
        else:
            dist = Lambda(self._diff_mult_dist)([gru_c, lstm_r])
            dist = Dense(1, activation='sigmoid', name="score_model")(dist)
        model = Model(context + [response], dist)
        return model

    def create_score_model(self) -> Model:
        cr = self.model.inputs
        if self.triplet_mode:
            emb_c = self.model.get_layer("gru").output
            emb_r = self.model.get_layer("pooling").get_output(-1)
            dist_score = Lambda(lambda x: self.euclidian_dist(x), name="score_model")
            score = dist_score([emb_c, emb_r])
        else:
            score = self.model.get_layer("score_model").output
            score = Lambda(lambda x: 1. - K.squeeze(x, -1))(score)
        score = Lambda(lambda x: 1. - x)(score)
        model = Model(cr, score)
        return model

    def create_context_model(self) -> Model:
        m = Model(self.model.inputs[:-1],
                  self.model.get_layer("gru").output)
        return m

    def create_response_model(self) -> Model:
        m = Model(self.model.inputs[-1],
                  self.model.get_layer("pooling").get_output_at(-1))
        return m
