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
from tensorflow.keras.initializers import glorot_uniform, Orthogonal
from tensorflow.keras.layers import Input, LSTM, Lambda, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Model

from deeppavlov.core.common.registry import register
from deeppavlov.core.layers.keras_layers import AttentiveMatchingLayer, MaxattentiveMatchingLayer
from deeppavlov.core.layers.keras_layers import FullMatchingLayer, MaxpoolingMatchingLayer
from deeppavlov.models.ranking.bilstm_siamese_network import BiLSTMSiameseNetwork

log = getLogger(__name__)


@register('mpm_nn')
class MPMSiameseNetwork(BiLSTMSiameseNetwork):
    """The class implementing a siamese neural network with bilateral multi-Perspective matching.

    The network architecture is based on https://arxiv.org/abs/1702.03814.
    
    Args:
        dense_dim: Dimensionality of the dense layer.
        perspective_num: Number of perspectives in multi-perspective matching layers.
        aggregation dim: Dimensionality of the hidden state in the second BiLSTM layer.
        inpdrop_val: Float between 0 and 1. A dropout value for the linear transformation of the inputs.
        recdrop_val: Float between 0 and 1. A dropout value for the linear transformation of the recurrent state.
        ldrop_val: A dropout value of the dropout layer before the second BiLSTM layer.
        dropout_val:  A dropout value of the dropout layer after the second BiLSTM layer.
    """

    def __init__(self,
                 dense_dim: int = 50,
                 perspective_num: int = 20,
                 aggregation_dim: int = 200,
                 recdrop_val: float = 0.0,
                 inpdrop_val: float = 0.0,
                 ldrop_val: float = 0.0,
                 dropout_val: float = 0.0,
                 *args,
                 **kwargs) -> None:

        self.dense_dim = dense_dim
        self.perspective_num = perspective_num
        self.aggregation_dim = aggregation_dim
        self.ldrop_val = ldrop_val
        self.recdrop_val = recdrop_val
        self.inpdrop_val = inpdrop_val
        self.dropout_val = dropout_val
        self.seed = kwargs.get("triplet_loss")
        self.triplet_mode = kwargs.get("triplet_loss")

        super(MPMSiameseNetwork, self).__init__(*args, **kwargs)

    def create_lstm_layer_1(self):
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        bioutp = Bidirectional(LSTM(self.hidden_dim,
                                    input_shape=(self.max_sequence_length, self.embedding_dim,),
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    recurrent_dropout=self.recdrop_val,
                                    dropout=self.inpdrop_val,
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=True), merge_mode=None)
        return bioutp

    def create_lstm_layer_2(self):
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        bioutp = Bidirectional(LSTM(self.aggregation_dim,
                                    input_shape=(self.max_sequence_length, 8 * self.perspective_num,),
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    recurrent_dropout=self.recdrop_val,
                                    dropout=self.inpdrop_val,
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=False),
                               merge_mode='concat',
                               name="sentence_embedding")
        return bioutp

    def create_model(self) -> Model:
        if self.use_matrix:
            context = Input(shape=(self.max_sequence_length,))
            response = Input(shape=(self.max_sequence_length,))
            emb_layer = self.embedding_layer()
            emb_c = emb_layer(context)
            emb_r = emb_layer(response)
        else:
            context = Input(shape=(self.max_sequence_length, self.embedding_dim,))
            response = Input(shape=(self.max_sequence_length, self.embedding_dim,))
            emb_c = context
            emb_r = response
        lstm_layer = self.create_lstm_layer_1()
        lstm_a = lstm_layer(emb_c)
        lstm_b = lstm_layer(emb_r)

        f_layer_f = FullMatchingLayer(self.perspective_num)
        f_layer_b = FullMatchingLayer(self.perspective_num)
        f_a_forw = f_layer_f([lstm_a[0], lstm_b[0]])[0]
        f_a_back = f_layer_b([Lambda(lambda x: K.reverse(x, 1))(lstm_a[1]),
                              Lambda(lambda x: K.reverse(x, 1))(lstm_b[1])])[0]
        f_a_back = Lambda(lambda x: K.reverse(x, 1))(f_a_back)
        f_b_forw = f_layer_f([lstm_b[0], lstm_a[0]])[0]
        f_b_back = f_layer_b([Lambda(lambda x: K.reverse(x, 1))(lstm_b[1]),
                              Lambda(lambda x: K.reverse(x, 1))(lstm_a[1])])[0]
        f_b_back = Lambda(lambda x: K.reverse(x, 1))(f_b_back)

        mp_layer_f = MaxpoolingMatchingLayer(self.perspective_num)
        mp_layer_b = MaxpoolingMatchingLayer(self.perspective_num)
        mp_a_forw = mp_layer_f([lstm_a[0], lstm_b[0]])[0]
        mp_a_back = mp_layer_b([lstm_a[1], lstm_b[1]])[0]
        mp_b_forw = mp_layer_f([lstm_b[0], lstm_a[0]])[0]
        mp_b_back = mp_layer_b([lstm_b[1], lstm_a[1]])[0]

        at_layer_f = AttentiveMatchingLayer(self.perspective_num)
        at_layer_b = AttentiveMatchingLayer(self.perspective_num)
        at_a_forw = at_layer_f([lstm_a[0], lstm_b[0]])[0]
        at_a_back = at_layer_b([lstm_a[1], lstm_b[1]])[0]
        at_b_forw = at_layer_f([lstm_b[0], lstm_a[0]])[0]
        at_b_back = at_layer_b([lstm_b[1], lstm_a[1]])[0]

        ma_layer_f = MaxattentiveMatchingLayer(self.perspective_num)
        ma_layer_b = MaxattentiveMatchingLayer(self.perspective_num)
        ma_a_forw = ma_layer_f([lstm_a[0], lstm_b[0]])[0]
        ma_a_back = ma_layer_b([lstm_a[1], lstm_b[1]])[0]
        ma_b_forw = ma_layer_f([lstm_b[0], lstm_a[0]])[0]
        ma_b_back = ma_layer_b([lstm_b[1], lstm_a[1]])[0]

        concat_a = Lambda(lambda x: K.concatenate(x, axis=-1))([f_a_forw, f_a_back,
                                                                mp_a_forw, mp_a_back,
                                                                at_a_forw, at_a_back,
                                                                ma_a_forw, ma_a_back])
        concat_b = Lambda(lambda x: K.concatenate(x, axis=-1))([f_b_forw, f_b_back,
                                                                mp_b_forw, mp_b_back,
                                                                at_b_forw, at_b_back,
                                                                ma_b_forw, ma_b_back])

        concat_a = Dropout(self.ldrop_val)(concat_a)
        concat_b = Dropout(self.ldrop_val)(concat_b)

        lstm_layer_agg = self.create_lstm_layer_2()
        agg_a = lstm_layer_agg(concat_a)
        agg_b = lstm_layer_agg(concat_b)

        agg_a = Dropout(self.dropout_val)(agg_a)
        agg_b = Dropout(self.dropout_val)(agg_b)

        reduced = Lambda(lambda x: K.concatenate(x, axis=-1))([agg_a, agg_b])

        if self.triplet_mode:
            dist = Lambda(self._pairwise_distances)([agg_a, agg_b])
        else:
            ker_in = glorot_uniform(seed=self.seed)
            dense = Dense(self.dense_dim, kernel_initializer=ker_in)(reduced)
            dist = Dense(1, activation='sigmoid', name="score_model")(dense)
        model = Model([context, response], dist)
        return model
