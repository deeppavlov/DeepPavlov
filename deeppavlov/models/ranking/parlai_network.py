import numpy as np
from keras.layers import Input, LSTM, Lambda, Dense, Dropout, Activation
from keras.layers.merge import multiply
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.initializers import glorot_uniform, Orthogonal
from keras import backend as K

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.keras_siamese_model import KerasSiameseModel

log = get_logger(__name__)


@register('parlai_nn')
class ParlaiNetwork(KerasSiameseModel):

    """The class implementing a siamese neural network with BiLSTM and max pooling.

    There is a possibility to use a binary cross-entropy loss as well as
    a triplet loss with random or hard negative sampling.

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

    def __init__(self,
                 len_vocab: int,
                 dense_dim: int = 50,
                 perspective_num: int = 10,
                 aggregation_dim: int = 4,
                 ldrop_val: float = 0.0,
                 recdrop_val: float = 0.0,
                 inpdrop_val: float = 0.0,
                 dropout_val: float = 0.0,
                 seed: int = None,
                 shared_weights: bool = True,
                 embedding_dim: int = 300,
                 reccurent: str = "bilstm",
                 hidden_dim: int = 300,
                 max_pooling: bool = True,
                 margin: float = 0.1,
                 *args,
                 **kwargs):
        self.dense_dim = dense_dim
        self.perspective_num = perspective_num
        self.aggregation_dim = aggregation_dim
        self.ldrop_val = ldrop_val
        self.recdrop_val = recdrop_val
        self.inpdrop_val = inpdrop_val
        self.dropout_val = dropout_val
        self.toks_num = len_vocab
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.shared_weights = shared_weights
        self.pooling = max_pooling
        self.recurrent = reccurent
        self.margin = margin
        self.embedding_dim = embedding_dim

        super(ParlaiNetwork, self).__init__(*args, **kwargs)

    def create_lstm_layer(self, input_dim):
        """Create a LSTM layer of a model."""

        inp = Input(shape=(input_dim, self.embedding_dim,))
        inp_dropout = Dropout(self.ldrop_val)(inp)
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        outp = LSTM(self.hidden_dim, input_shape=(input_dim, self.embedding_dim,),
                    kernel_regularizer=None,
                    recurrent_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    recurrent_dropout=self.recdrop_val,
                    dropout=self.inpdrop_val,
                    kernel_initializer=ker_in,
                    recurrent_initializer=rec_in,
                    return_sequences=True)(inp_dropout)
        outp_dropout = Dropout(self.dropout_val)(outp)
        model = Model(inputs=inp, outputs=outp_dropout, name="LSTM_encoder")
        return model

    def create_lstm_layer_1(self, input_dim):
        """Create a LSTM layer of a model."""

        inp = Input(shape=(input_dim,  self.embedding_dim,))
        inp_drop = Dropout(self.ldrop_val)(inp)
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        bioutp = Bidirectional(LSTM(self.hidden_dim,
                                    input_shape=(input_dim, self.embedding_dim,),
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    recurrent_dropout=self.recdrop_val,
                                    dropout=self.inpdrop_val,
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=True), merge_mode=None)(inp_drop)
        dropout_forw = Dropout(self.dropout_val)(bioutp[0])
        dropout_back = Dropout(self.dropout_val)(bioutp[1])
        model = Model(inputs=inp, outputs=[dropout_forw, dropout_back], name="biLSTM_encoder")
        return model

    def create_lstm_layer_2(self, input_dim):
        """Create a LSTM layer of a model."""

        inp = Input(shape=(input_dim, 2*self.perspective_num,))
        inp_drop = Dropout(self.ldrop_val)(inp)
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        bioutp = Bidirectional(LSTM(self.aggregation_dim,
                                    input_shape=(input_dim, 2*self.perspective_num,),
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    recurrent_dropout=self.recdrop_val,
                                    dropout=self.inpdrop_val,
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=True), merge_mode=None)(inp_drop)
        dropout_forw = Dropout(self.dropout_val)(bioutp[0])
        dropout_back = Dropout(self.dropout_val)(bioutp[1])
        model = Model(inputs=inp, outputs=[dropout_forw, dropout_back], name="biLSTM_enc_persp")
        return model

    def create_lstm_layer_last(self, input_dim):
        """Create a LSTM layer of a model."""

        inp = Input(shape=(input_dim,  self.embedding_dim,))
        inp_drop = Dropout(self.ldrop_val)(inp)
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        bioutp = Bidirectional(LSTM(self.hidden_dim,
                                    input_shape=(input_dim, self.embedding_dim,),
                                    kernel_regularizer=None,
                                    recurrent_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None,
                                    recurrent_dropout=self.recdrop_val,
                                    dropout=self.inpdrop_val,
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=False), merge_mode='concat')(inp_drop)
        dropout = Dropout(self.dropout_val)(bioutp)
        model = Model(inputs=inp, outputs=dropout, name="biLSTM_encoder_last")
        return model

    def create_attention_layer(self, input_dim_a, input_dim_b):
        """Create an attention layer of a model."""

        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        val = np.concatenate((np.zeros((self.max_sequence_length-1,1)), np.ones((1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp_b_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp_b)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp_b_perm)
        ker_in = glorot_uniform(seed=self.seed)
        outp_a = Dense(self.attention_dim, input_shape=(input_dim_a, self.hidden_dim),
                       kernel_initializer=ker_in, activation='relu')(inp_a)
        outp_last = Dense(self.attention_dim, input_shape=(1, self.hidden_dim),
                          kernel_initializer=ker_in, activation='relu')(last_state)
        outp_last_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_last)
        outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_last_perm, outp_a])
        outp_norm = Activation('softmax')(outp)
        outp_norm_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_norm)
        model = Model(inputs=[inp_a, inp_b], outputs=outp_norm_perm, name="attention_generator")
        return model

    def create_attention_layer_f(self, input_dim_a, input_dim_b):
        """Create an attention layer of a model."""

        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        val = np.concatenate((np.zeros((self.max_sequence_length-1,1)), np.ones((1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp_b_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp_b)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp_b_perm)
        ker_in = glorot_uniform(seed=self.seed)
        outp_a = Dense(self.attention_dim, input_shape=(input_dim_a, self.hidden_dim),
                       kernel_initializer=ker_in, activation='relu')(inp_a)
        outp_last = Dense(self.attention_dim, input_shape=(1, self.hidden_dim),
                          kernel_initializer=ker_in, activation='relu')(last_state)
        outp_last_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_last)
        outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_last_perm, outp_a])
        outp_norm = Activation('softmax')(outp)
        outp_norm_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_norm)
        model = Model(inputs=[inp_a, inp_b], outputs=outp_norm_perm, name="att_generator_forw")
        return model

    def create_attention_layer_b(self, input_dim_a, input_dim_b):
        """Create an attention layer of a model."""

        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        val = np.concatenate((np.ones((1,1)), np.zeros((self.max_sequence_length-1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp_b_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp_b)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp_b_perm)
        ker_in = glorot_uniform(seed=self.seed)
        outp_a = Dense(self.attention_dim, input_shape=(input_dim_a, self.hidden_dim),
                       kernel_initializer=ker_in, activation='relu')(inp_a)
        outp_last = Dense(self.attention_dim, input_shape=(1, self.hidden_dim),
                          kernel_initializer=ker_in, activation='relu')(last_state)
        outp_last_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_last)
        outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_last_perm, outp_a])
        outp_norm = Activation('softmax')(outp)
        outp_norm_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_norm)
        model = Model(inputs=[inp_a, inp_b], outputs=outp_norm_perm, name="att_generator_back")
        return model

    def weighted_with_attention(self, inputs):
        """Define a function for a lambda layer of a model."""

        inp, inp_cont = inputs
        val = np.eye(self.max_sequence_length)
        kcon = K.constant(value=val, dtype='float32')
        diag = K.repeat_elements(inp_cont, self.max_sequence_length, 2) * kcon
        return K.batch_dot(diag, K.permute_dimensions(inp, (0,2,1)), axes=[1,2])

    def weighted_with_attention_output_shape(self, shapes):
        """Define an output shape of a lambda layer of a model."""

        shape1, shape2 = shapes
        return shape1

    def dim_reduction(self, inp):
        """Define a function for a lambda layer of a model."""

        return K.sum(inp, axis=1)

    def dim_reduction_output_shape(self, shape):
        """Define an output shape of a lambda layer of a model."""

        return shape[0], shape[2]

    def weight_and_reduce(self, inputs):
        """Define a function for a lambda layer of a model."""

        inp, inp_cont = inputs
        reduced = K.batch_dot(inp_cont,
                              K.permute_dimensions(inp, (0,2,1)), axes=[1,2])
        return K.squeeze(reduced, 1)

    def weight_and_reduce_output_shape(self, shapes):
        """Define an output shape of a lambda layer of a model."""

        shape1, shape2 = shapes
        return shape1[0], shape1[2]

    def cosine_dist(self, inputs):
        """Define a function for a lambda layer of a model."""

        input1, input2 = inputs
        a = K.abs(input1-input2)
        b = multiply(inputs)
        return K.concatenate([a, b])

    def cosine_dist_output_shape(self, shapes):
        """Define an output shape of a lambda layer of a model."""

        shape1, shape2 = shapes
        return shape1[0], 2*shape1[1]

    def create_full_matching_layer_f(self, input_dim_a, input_dim_b):
        """Create a full-matching layer of a model."""

        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        W = []
        for i in range(self.perspective_num):
            wi = K.random_uniform_variable((1, self.hidden_dim), -1.0, 1.0,
                                           seed=self.seed if self.seed is not None else 243)
            W.append(wi)

        val = np.concatenate((np.zeros((self.max_sequence_length-1,1)), np.ones((1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp_b_perm = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp_b)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp_b_perm)
        m = []
        for i in range(self.perspective_num):
            outp_a = Lambda(lambda x: x * W[i])(inp_a)
            outp_last = Lambda(lambda x: x * W[i])(last_state)
            outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(outp_a)
            outp_last = Lambda(lambda x: K.l2_normalize(x, -1))(outp_last)
            outp_last = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_last)
            outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_last, outp_a])
            outp = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp)
            m.append(outp)
        if self.perspective_num > 1:
            persp = Lambda(lambda x: K.concatenate(x, 2))(m)
        else:
            persp = m
        model = Model(inputs=[inp_a, inp_b], outputs=persp)
        return model

    def create_full_matching_layer_b(self, input_dim_a, input_dim_b):
        """Create a full-matching layer of a model."""

        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        W = []
        for i in range(self.perspective_num):
            wi = K.random_uniform_variable((1, self.hidden_dim), -1.0, 1.0,
                                           seed=self.seed if self.seed is not None else 243)
            W.append(wi)

        val = np.concatenate((np.ones((1, 1)), np.zeros((self.max_sequence_length - 1, 1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp_b_perm = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(inp_b)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0, 2, 1)))(inp_b_perm)
        m = []
        for i in range(self.perspective_num):
            outp_a = Lambda(lambda x: x * W[i])(inp_a)
            outp_last = Lambda(lambda x: x * W[i])(last_state)
            outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(outp_a)
            outp_last = Lambda(lambda x: K.l2_normalize(x, -1))(outp_last)
            outp_last = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp_last)
            outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_last, outp_a])
            outp = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp)
            m.append(outp)
        if self.perspective_num > 1:
            persp = Lambda(lambda x: K.concatenate(x, 2))(m)
        else:
            persp = m
        model = Model(inputs=[inp_a, inp_b], outputs=persp)
        return model

    def create_maxpool_matching_layer(self, input_dim_a, input_dim_b):
        """Create a maxpooling-matching layer of a model."""

        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))
        W = []
        for i in range(self.perspective_num):
            wi = K.random_uniform_variable((1, self.hidden_dim), -1.0, 1.0,
                                           seed=self.seed if self.seed is not None else 243)
            W.append(wi)

        m = []
        for i in range(self.perspective_num):
            outp_a = Lambda(lambda x: x * W[i])(inp_a)
            outp_b = Lambda(lambda x: x * W[i])(inp_b)
            outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(outp_a)
            outp_b = Lambda(lambda x: K.l2_normalize(x, -1))(outp_b)
            outp_b = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp_b)
            outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_b, outp_a])
            outp = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(outp)
            outp = Lambda(lambda x: K.max(x, -1, keepdims=True))(outp)
            m.append(outp)
        if self.perspective_num > 1:
            persp = Lambda(lambda x: K.concatenate(x, 2))(m)
        else:
            persp = m
        model = Model(inputs=[inp_a, inp_b], outputs=persp)
        return model

    def create_att_matching_layer(self, input_dim_a, input_dim_b):
        """Create an attentive-matching layer of a model."""

        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))

        w = []
        for i in range(self.perspective_num):
            wi = K.random_uniform_variable((1, self.hidden_dim), -1.0, 1.0,
                                           seed=self.seed if self.seed is not None else 243)
            w.append(wi)

        outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(inp_a)
        outp_b = Lambda(lambda x: K.l2_normalize(x, -1))(inp_b)
        outp_b = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp_b)
        alpha = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_b, outp_a])
        alpha = Lambda(lambda x: K.l2_normalize(x, 1))(alpha)
        hmean = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([alpha, outp_b])

        m = []
        for i in range(self.perspective_num):
            outp_a = Lambda(lambda x: x * w[i])(inp_a)
            outp_hmean = Lambda(lambda x: x * w[i])(hmean)
            outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(outp_a)
            outp_hmean = Lambda(lambda x: K.l2_normalize(x, -1))(outp_hmean)
            outp_hmean = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp_hmean)
            outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_hmean, outp_a])
            val = np.eye(self.max_sequence_length)
            kcon = K.constant(value=val, dtype='float32')
            outp = Lambda(lambda x: K.sum(x * kcon, -1, keepdims=True))(outp)
            m.append(outp)
        if self.perspective_num > 1:
            persp = Lambda(lambda x: K.concatenate(x, 2))(m)
        else:
            persp = m
        model = Model(inputs=[inp_a, inp_b], outputs=persp)
        return model

    def create_maxatt_matching_layer(self, input_dim_a, input_dim_b):
        """Create a max-attentive-matching layer of a model."""

        inp_a = Input(shape=(input_dim_a, self.hidden_dim,))
        inp_b = Input(shape=(input_dim_b, self.hidden_dim,))

        W = []
        for i in range(self.perspective_num):
            wi = K.random_uniform_variable((1, self.hidden_dim), -1.0, 1.0,
                                           seed=self.seed if self.seed is not None else 243)
            W.append(wi)

        outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(inp_a)
        outp_b = Lambda(lambda x: K.l2_normalize(x, -1))(inp_b)
        outp_b = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp_b)
        alpha = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_b, outp_a])
        alpha = Lambda(lambda x: K.one_hot(K.argmax(x, 1), self.max_sequence_length))(alpha)
        hmax = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([alpha, outp_b])

        m = []
        for i in range(self.perspective_num):
            outp_a = Lambda(lambda x: x * W[i])(inp_a)
            outp_hmax = Lambda(lambda x: x * W[i])(hmax)
            outp_a = Lambda(lambda x: K.l2_normalize(x, -1))(outp_a)
            outp_hmax = Lambda(lambda x: K.l2_normalize(x, -1))(outp_hmax)
            outp_hmax = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(outp_hmax)
            outp = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]))([outp_hmax, outp_a])
            val = np.eye(self.max_sequence_length)
            kcon = K.constant(value=val, dtype='float32')
            outp = Lambda(lambda x: K.sum(x * kcon, -1, keepdims=True))(outp)
            m.append(outp)
        if self.perspective_num > 1:
            persp = Lambda(lambda x: K.concatenate(x, 2))(m)
        else:
            persp = m
        model = Model(inputs=[inp_a, inp_b], outputs=persp)
        return model

    def cosine_dist(self, inputs):
        input1, input2 = inputs
        a = K.abs(input1-input2)
        b = multiply(inputs)
        return K.concatenate([a, b])

    def cosine_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return shape1[0], 2*shape1[1]

    def terminal_f(self, inp):
        val = np.concatenate((np.zeros((self.max_sequence_length-1,1)), np.ones((1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp)
        return K.squeeze(last_state, 1)

    def terminal_f_output_shape(self, shape):
        return shape[0], shape[2]

    def terminal_b(self, inp):
        val = np.concatenate((np.ones((1,1)), np.zeros((self.max_sequence_length-1,1))), axis=0)
        kcon = K.constant(value=val, dtype='float32')
        inp = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(inp)
        last_state = Lambda(lambda x: K.permute_dimensions(K.dot(x, kcon), (0,2,1)))(inp)
        return K.squeeze(last_state, 1)

    def terminal_b_output_shape(self, shape):
        return shape[0], shape[2]

    def bmwacor_model(self):
        """Define a model with LSTM layers and with attention."""

        input_a = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        input_b = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        lstm_layer = self.create_lstm_layer(self.max_sequence_length)
        lstm_a = lstm_layer(input_a)
        lstm_b = lstm_layer(input_b)

        attention_layer = self.create_attention_layer(self.max_sequence_length, self.max_sequence_length)
        attention_a = attention_layer([lstm_a, lstm_b])
        attention_b = attention_layer([lstm_b, lstm_a])

        reduced_a = Lambda(self.weight_and_reduce,
                           output_shape=self.weight_and_reduce_output_shape, name="mul_sum_q1")([lstm_a, attention_a])
        reduced_b = Lambda(self.weight_and_reduce,
                           output_shape=self.weight_and_reduce_output_shape, name="mul_sum_q2")([lstm_b, attention_b])

        dist = Lambda(self.cosine_dist, output_shape=self.cosine_dist_output_shape,
                      name="similarity_network")([reduced_a, reduced_b])

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dist)

        model = Model([input_a, input_b], dense)

        return model

    def bilstm_split_model(self):
        """Define a model with bi-LSTM layers and with attention."""

        input_a = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        input_b = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        lstm_a = lstm_layer(input_a)
        lstm_b = lstm_layer(input_b)

        attention_layer_f = self.create_attention_layer_f(self.max_sequence_length, self.max_sequence_length)
        attention_layer_b = self.create_attention_layer_b(self.max_sequence_length, self.max_sequence_length)
        attention_a_forw = attention_layer_f([lstm_a[0], lstm_b[0]])
        attention_a_back = attention_layer_b([lstm_a[1], lstm_b[1]])
        attention_b_forw = attention_layer_f([lstm_b[0], lstm_a[0]])
        attention_b_back = attention_layer_b([lstm_b[1], lstm_a[1]])

        reduced_a_forw = Lambda(self.weight_and_reduce,
                                output_shape=self.weight_and_reduce_output_shape,
                                name="sum_q1_f")([lstm_a[0], attention_a_forw])
        reduced_a_back = Lambda(self.weight_and_reduce,
                                output_shape=self.weight_and_reduce_output_shape,
                                name="sum_q1_b")([lstm_a[1], attention_a_back])
        reduced_b_forw = Lambda(self.weight_and_reduce,
                                output_shape=self.weight_and_reduce_output_shape,
                                name="sum_q2_f")([lstm_b[0], attention_b_forw])
        reduced_b_back = Lambda(self.weight_and_reduce,
                                output_shape=self.weight_and_reduce_output_shape,
                                name="sum_q2_b")([lstm_b[1], attention_b_back])

        reduced_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                           name='concat_q1')([reduced_a_forw, reduced_a_back])
        reduced_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                           name='concat_q2')([reduced_b_forw, reduced_b_back])

        dist = Lambda(self.cosine_dist, output_shape=self.cosine_dist_output_shape,
                      name="similarity_network")([reduced_a, reduced_b])

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dist)

        model = Model([input_a, input_b], dense)

        return model

    def maxpool_match_model(self):
        """Define a model with maxpooling-matching layers."""

        input_a = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        input_b = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        lstm_a = lstm_layer(input_a)
        lstm_b = lstm_layer(input_b)

        matching_layer_f = self.create_maxpool_matching_layer(self.max_sequence_length, self.max_sequence_length)
        matching_layer_b = self.create_maxpool_matching_layer(self.max_sequence_length, self.max_sequence_length)
        lstm_layer_agg = self.create_lstm_layer_2(self.max_sequence_length)
        matching_a_forw = matching_layer_f([lstm_a[0], lstm_b[0]])
        matching_a_back = matching_layer_b([lstm_a[1], lstm_b[1]])
        matching_b_forw = matching_layer_f([lstm_b[0], lstm_a[0]])
        matching_b_back = matching_layer_b([lstm_b[1], lstm_a[1]])

        concat_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q1_match')([matching_a_forw, matching_a_back])
        concat_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q2_match')([matching_b_forw, matching_b_back])

        agg_a = lstm_layer_agg(concat_a)
        agg_b = lstm_layer_agg(concat_b)

        reduced_a_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q1_f")(agg_a[0])
        reduced_a_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q1_b")(agg_a[1])
        reduced_b_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q2_f")(agg_b[0])
        reduced_b_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q2_b")(agg_b[1])

        reduced = Lambda(lambda x: K.concatenate(x, axis=-1),
                         name='conc_agg')([reduced_a_forw, reduced_a_back,
                                           reduced_b_forw, reduced_b_back])

        ker_in = glorot_uniform(seed=self.seed)
        dense = Dense(self.dense_dim, kernel_initializer=ker_in)(reduced)

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dense)

        model = Model([input_a, input_b], dense)
        return model

    def maxatt_match_model(self):
        """Define a model with max-attentive-matching layers."""

        input_a = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        input_b = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        lstm_a = lstm_layer(input_a)
        lstm_b = lstm_layer(input_b)

        matching_layer_f = self.create_maxatt_matching_layer(self.max_sequence_length, self.max_sequence_length)
        matching_layer_b = self.create_maxatt_matching_layer(self.max_sequence_length, self.max_sequence_length)
        lstm_layer_agg = self.create_lstm_layer_2(self.max_sequence_length)
        matching_a_forw = matching_layer_f([lstm_a[0], lstm_b[0]])
        matching_a_back = matching_layer_b([lstm_a[1], lstm_b[1]])
        matching_b_forw = matching_layer_f([lstm_b[0], lstm_a[0]])
        matching_b_back = matching_layer_b([lstm_b[1], lstm_a[1]])

        concat_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q1_match')([matching_a_forw, matching_a_back])
        concat_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q2_match')([matching_b_forw, matching_b_back])

        agg_a = lstm_layer_agg(concat_a)
        agg_b = lstm_layer_agg(concat_b)

        reduced_a_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q1_f")(agg_a[0])
        reduced_a_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q1_b")(agg_a[1])
        reduced_b_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q2_f")(agg_b[0])
        reduced_b_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q2_b")(agg_b[1])

        reduced = Lambda(lambda x: K.concatenate(x, axis=-1),
                         name='conc_agg')([reduced_a_forw, reduced_a_back,
                                           reduced_b_forw, reduced_b_back])

        ker_in = glorot_uniform(seed=self.seed)
        dense = Dense(self.dense_dim, kernel_initializer=ker_in)(reduced)

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dense)

        model = Model([input_a, input_b], dense)
        return model

    def att_match_model(self):
        """Define a model with attentive-matching layers."""

        input_a = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        input_b = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        lstm_a = lstm_layer(input_a)
        lstm_b = lstm_layer(input_b)

        matching_layer_f = self.create_att_matching_layer(self.max_sequence_length, self.max_sequence_length)
        matching_layer_b = self.create_att_matching_layer(self.max_sequence_length, self.max_sequence_length)
        lstm_layer_agg = self.create_lstm_layer_2(self.max_sequence_length)
        matching_a_forw = matching_layer_f([lstm_a[0], lstm_b[0]])
        matching_a_back = matching_layer_b([lstm_a[1], lstm_b[1]])
        matching_b_forw = matching_layer_f([lstm_b[0], lstm_a[0]])
        matching_b_back = matching_layer_b([lstm_b[1], lstm_a[1]])

        concat_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q1_match')([matching_a_forw, matching_a_back])
        concat_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q2_match')([matching_b_forw, matching_b_back])

        agg_a = lstm_layer_agg(concat_a)
        agg_b = lstm_layer_agg(concat_b)

        reduced_a_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q1_f")(agg_a[0])
        reduced_a_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q1_b")(agg_a[1])
        reduced_b_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q2_f")(agg_b[0])
        reduced_b_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q2_b")(agg_b[1])

        reduced = Lambda(lambda x: K.concatenate(x, axis=-1),
                         name='conc_agg')([reduced_a_forw, reduced_a_back,
                                           reduced_b_forw, reduced_b_back])

        ker_in = glorot_uniform(seed=self.seed)
        dense = Dense(self.dense_dim, kernel_initializer=ker_in)(reduced)

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dense)

        model = Model([input_a, input_b], dense)
        return model

    def full_match_model(self):
        """Define a model with full-matching layers."""

        input_a = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        input_b = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        lstm_layer = self.create_lstm_layer_1(self.max_sequence_length)
        lstm_a = lstm_layer(input_a)
        lstm_b = lstm_layer(input_b)

        matching_layer_f = self.create_full_matching_layer_f(self.max_sequence_length, self.max_sequence_length)
        matching_layer_b = self.create_full_matching_layer_b(self.max_sequence_length, self.max_sequence_length)
        lstm_layer_agg = self.create_lstm_layer_2(self.max_sequence_length)
        matching_a_forw = matching_layer_f([lstm_a[0], lstm_b[0]])
        matching_a_back = matching_layer_b([lstm_a[1], lstm_b[1]])
        matching_b_forw = matching_layer_f([lstm_b[0], lstm_a[0]])
        matching_b_back = matching_layer_b([lstm_b[1], lstm_a[1]])

        concat_a = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q1_match')([matching_a_forw, matching_a_back])
        concat_b = Lambda(lambda x: K.concatenate(x, axis=-1),
                          name='conc_q2_match')([matching_b_forw, matching_b_back])

        agg_a = lstm_layer_agg(concat_a)
        agg_b = lstm_layer_agg(concat_b)

        reduced_a_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q1_f")(agg_a[0])
        reduced_a_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q1_b")(agg_a[1])
        reduced_b_forw = Lambda(self.terminal_f,
                                output_shape=self.terminal_f_output_shape, name="last_q2_f")(agg_b[0])
        reduced_b_back = Lambda(self.terminal_b,
                                output_shape=self.terminal_b_output_shape, name="last_q2_b")(agg_b[1])

        reduced = Lambda(lambda x: K.concatenate(x, axis=-1),
                         name='conc_agg')([reduced_a_forw, reduced_a_back,
                                           reduced_b_forw, reduced_b_back])

        ker_in = glorot_uniform(seed=self.seed)
        dense = Dense(self.dense_dim, kernel_initializer=ker_in)(reduced)

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dense)

        model = Model([input_a, input_b], dense)
        return model

    def bilstm_woatt_model(self):
        """Define a model with bi-LSTM layers and without attention."""

        input_a = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        input_b = Input(shape=(self.max_sequence_length, self.embedding_dim,))
        lstm_layer = self.create_lstm_layer_last(self.max_sequence_length)
        lstm_last_a = lstm_layer(input_a)
        lstm_last_b = lstm_layer(input_b)

        dist = Lambda(self.cosine_dist, output_shape=self.cosine_dist_output_shape,
                      name="similarity_network")([lstm_last_a, lstm_last_b])

        dense = Dense(1, activation='sigmoid', name='similarity_score',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None)(dist)

        model = Model([input_a, input_b], dense)

        return model

    def create_model(self):
        return self.full_match_model()