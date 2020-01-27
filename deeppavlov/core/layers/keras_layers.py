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

from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dense, Reshape, Concatenate, Lambda, Layer, Multiply


def expand_tile(units, axis):
    """
    Expand and tile tensor along given axis

    Args:
        units: tf tensor with dimensions [batch_size, time_steps, n_input_features]
        axis: axis along which expand and tile. Must be 1 or 2

    """
    assert axis in (1, 2)
    n_time_steps = K.int_shape(units)[1]
    repetitions = [1, 1, 1, 1]
    repetitions[axis] = n_time_steps
    if axis == 1:
        expanded = Reshape(target_shape=((1,) + K.int_shape(units)[1:]))(units)
    else:
        expanded = Reshape(target_shape=(K.int_shape(units)[1:2] + (1,) + K.int_shape(units)[2:]))(units)
    return K.tile(expanded, repetitions)


def additive_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """
    Compute additive self attention for time series of vectors (with batch dimension)
            the formula: score(h_i, h_j) = <v, tanh(W_1 h_i + W_2 h_j)>
            v is a learnable vector of n_hidden dimensionality,
            W_1 and W_2 are learnable [n_hidden, n_input_features] matrices

    Args:
        units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
        n_hidden: number of2784131 units in hidden representation of similarity measure
        n_output_features: number of features in output dense layer
        activation: activation at the output

    Returns:
        output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
        """
    n_input_features = K.int_shape(units)[2]
    if n_hidden is None:
        n_hidden = n_input_features
    if n_output_features is None:
        n_output_features = n_input_features
    exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
    exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
    units_pairs = Concatenate(axis=3)([exp1, exp2])
    query = Dense(n_hidden, activation="tanh")(units_pairs)
    attention = Dense(1, activation=lambda x: softmax(x, axis=2))(query)
    attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)
    output = Dense(n_output_features, activation=activation)(attended_units)
    return output


def multiplicative_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """
    Compute multiplicative self attention for time series of vectors (with batch dimension)
    the formula: score(h_i, h_j) = <W_1 h_i,  W_2 h_j>,  W_1 and W_2 are learnable matrices
    with dimensionality [n_hidden, n_input_features]

    Args:
        units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
        n_hidden: number of units in hidden representation of similarity measure
        n_output_features: number of features in output dense layer
        activation: activation at the output

    Returns:
        output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
    """
    n_input_features = K.int_shape(units)[2]
    if n_hidden is None:
        n_hidden = n_input_features
    if n_output_features is None:
        n_output_features = n_input_features
    exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
    exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
    queries = Dense(n_hidden)(exp1)
    keys = Dense(n_hidden)(exp2)
    scores = Lambda(lambda x: K.sum(queries * x, axis=3, keepdims=True))(keys)
    attention = Lambda(lambda x: softmax(x, axis=2))(scores)
    mult = Multiply()([attention, exp1])
    attended_units = Lambda(lambda x: K.sum(x, axis=2))(mult)
    output = Dense(n_output_features, activation=activation)(attended_units)
    return output


class MatchingLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.W = []
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W = []
        for i in range(self.output_dim):
            self.W.append(self.add_weight(name='kernel',
                                          shape=(1, input_shape[0][-1]),
                                          initializer='uniform',
                                          trainable=True))
        super().build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], shape_a[1], self.output_dim), (shape_a[0], shape_a[1], self.output_dim)]


class FullMatchingLayer(MatchingLayer):

    def call(self, x, **kwargs):
        assert isinstance(x, list)
        inp_a, inp_b = x
        last_state = K.expand_dims(inp_b[:, -1, :], 1)
        m = []
        for i in range(self.output_dim):
            outp_a = inp_a * self.W[i]
            outp_last = last_state * self.W[i]
            outp_a = K.l2_normalize(outp_a, -1)
            outp_last = K.l2_normalize(outp_last, -1)
            outp = K.batch_dot(outp_a, outp_last, axes=[2, 2])
            m.append(outp)
        if self.output_dim > 1:
            persp = K.concatenate(m, 2)
        else:
            persp = m[0]
        return [persp, persp]


class MaxpoolingMatchingLayer(MatchingLayer):

    def call(self, x, **kwargs):
        assert isinstance(x, list)
        inp_a, inp_b = x
        m = []
        for i in range(self.output_dim):
            outp_a = inp_a * self.W[i]
            outp_b = inp_b * self.W[i]
            outp_a = K.l2_normalize(outp_a, -1)
            outp_b = K.l2_normalize(outp_b, -1)
            outp = K.batch_dot(outp_a, outp_b, axes=[2, 2])
            outp = K.max(outp, -1, keepdims=True)
            m.append(outp)
        if self.output_dim > 1:
            persp = K.concatenate(m, 2)
        else:
            persp = m[0]
        return [persp, persp]


class AttentiveMatchingLayer(MatchingLayer):

    def call(self, x, **kwargs):
        assert isinstance(x, list)
        inp_a, inp_b = x

        outp_a = K.l2_normalize(inp_a, -1)
        outp_b = K.l2_normalize(inp_b, -1)
        alpha = K.batch_dot(outp_b, outp_a, axes=[1, 1])
        alpha = K.l2_normalize(alpha, 1)
        hmean = K.batch_dot(outp_b, alpha, axes=[2, 1])
        kcon = K.eye(K.int_shape(inp_a)[1], dtype='float32')

        m = []
        for i in range(self.output_dim):
            outp_a = inp_a * self.W[i]
            outp_hmean = hmean * self.W[i]
            outp_a = K.l2_normalize(outp_a, -1)
            outp_hmean = K.l2_normalize(outp_hmean, -1)
            outp = K.batch_dot(outp_hmean, outp_a, axes=[2, 2])
            outp = K.sum(outp * kcon, -1, keepdims=True)
            m.append(outp)
        if self.output_dim > 1:
            persp = K.concatenate(m, 2)
        else:
            persp = m[0]
        return [persp, persp]


class MaxattentiveMatchingLayer(MatchingLayer):

    def call(self, x, **kwargs):
        assert isinstance(x, list)
        inp_a, inp_b = x

        outp_a = K.l2_normalize(inp_a, -1)
        outp_b = K.l2_normalize(inp_b, -1)
        alpha = K.batch_dot(outp_b, outp_a, axes=[2, 2])
        alpha = K.l2_normalize(alpha, 1)
        alpha = K.one_hot(K.argmax(alpha, 1), K.int_shape(inp_a)[1])
        hmax = K.batch_dot(alpha, outp_b, axes=[1, 1])
        kcon = K.eye(K.int_shape(inp_a)[1], dtype='float32')

        m = []
        for i in range(self.output_dim):
            outp_a = inp_a * self.W[i]
            outp_hmax = hmax * self.W[i]
            outp_a = K.l2_normalize(outp_a, -1)
            outp_hmax = K.l2_normalize(outp_hmax, -1)
            outp = K.batch_dot(outp_hmax, outp_a, axes=[2, 2])
            outp = K.sum(outp * kcon, -1, keepdims=True)
            m.append(outp)
        if self.output_dim > 1:
            persp = K.concatenate(m, 2)
        else:
            persp = m[0]
        return [persp, persp]
