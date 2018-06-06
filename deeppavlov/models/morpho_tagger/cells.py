import numpy as np

import keras.backend as kb
import keras.layers as kl
import keras.activations as kact
import keras.initializers as kinit
from keras.engine.topology import InputSpec


INFTY = -100


class Highway(kl.Layer):

    def __init__(self, activation=None, bias_initializer=-1, **kwargs):
        super(Highway, self).__init__(**kwargs)
        self.activation = kact.get(activation)
        self.bias_initializer = bias_initializer
        if isinstance(self.bias_initializer, int):
            self.bias_initializer = kinit.constant(self.bias_initializer)
        self.input_spec = [InputSpec(min_ndim=2)]

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.gate_kernel = self.add_weight(
            shape=(input_dim, input_dim), initializer='uniform', name='gate_kernel')
        self.gate_bias = self.add_weight(
            shape=(input_dim,), initializer=self.bias_initializer, name='gate_bias')
        self.dense_kernel = self.add_weight(
            shape=(input_dim, input_dim), initializer='uniform', name='dense_kernel')
        self.dense_bias = self.add_weight(
            shape=(input_dim,), initializer=self.bias_initializer, name='dense_bias')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        gate = kb.dot(inputs, self.gate_kernel)
        gate = kb.bias_add(gate, self.gate_bias, data_format="channels_last")
        gate = self.activation(gate)
        new_value = kb.dot(inputs, self.dense_kernel)
        new_value = kb.bias_add(new_value, self.dense_bias, data_format="channels_last")
        return gate * new_value + (1.0 - gate) * inputs

    def compute_output_shape(self, input_shape):
        return input_shape


def weighted_sum(first, second, sigma, first_threshold=-np.inf, second_threshold=np.inf):
    logit_probs = first * sigma + second * (1.0 - sigma)
    infty_tensor = kb.ones_like(logit_probs) * INFTY
    logit_probs = kb.switch(kb.greater(first, first_threshold), logit_probs, infty_tensor)
    logit_probs = kb.switch(kb.greater(second, second_threshold), logit_probs, infty_tensor)
    return logit_probs


class WeightedCombinationLayer(kl.Layer):

    """
    A class for weighted combination of probability distributions
    """

    def __init__(self, first_threshold=None, second_threshold=None,
                 use_dimension_bias=False, use_intermediate_layer=False,
                 intermediate_dim=64, intermediate_activation=None,
                 from_logits=False, return_logits=False,
                 bias_initializer=1.0, **kwargs):
        # if 'input_shape' not in kwargs:
        #     kwargs['input_shape'] = [(None, input_dim,), (None, input_dim)]
        super(WeightedCombinationLayer, self).__init__(**kwargs)
        self.first_threshold = first_threshold if first_threshold is not None else INFTY
        self.second_threshold = second_threshold if second_threshold is not None else INFTY
        self.use_dimension_bias = use_dimension_bias
        self.use_intermediate_layer = use_intermediate_layer
        self.intermediate_dim = intermediate_dim
        self.intermediate_activation = kact.get(intermediate_activation)
        self.from_logits = from_logits
        self.return_logits = return_logits
        self.bias_initializer = bias_initializer
        self.input_spec = [InputSpec(), InputSpec(), InputSpec()]

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[0] == input_shape[1]
        assert input_shape[0][:-1] == input_shape[2][:-1]

        input_dim, features_dim = input_shape[0][-1], input_shape[2][-1]
        if self.use_intermediate_layer:
            self.first_kernel = self.add_weight(
                shape=(features_dim, self.intermediate_dim),
                initializer="random_uniform", name='first_kernel')
            self.first_bias = self.add_weight(
                shape=(self.intermediate_dim,),
                initializer="random_uniform", name='first_bias')
        self.features_kernel = self.add_weight(
            shape=(features_dim, 1), initializer="random_uniform", name='kernel')
        self.features_bias = self.add_weight(
            shape=(1,), initializer=kinit.Constant(self.bias_initializer), name='bias')
        if self.use_dimension_bias:
            self.dimensions_bias = self.add_weight(
                shape=(input_dim,), initializer="random_uniform", name='dimension_bias')
        super(WeightedCombinationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list) and len(inputs) == 3
        first, second, features = inputs[0], inputs[1], inputs[2]
        if not self.from_logits:
            first = kb.clip(first, 1e-10, 1.0)
            second = kb.clip(second, 1e-10, 1.0)
            first_, second_ = kb.log(first), kb.log(second)
        else:
            first_, second_ = first, second
        # embedded_features.shape = (M, T, 1)
        if self.use_intermediate_layer:
            features = kb.dot(features, self.first_kernel)
            features = kb.bias_add(features, self.first_bias, data_format="channels_last")
            features = self.intermediate_activation(features)
        embedded_features = kb.dot(features, self.features_kernel)
        embedded_features = kb.bias_add(
            embedded_features, self.features_bias, data_format="channels_last")
        if self.use_dimension_bias:
            tiling_shape = [1] * (kb.ndim(first)-1) + [kb.shape(first)[-1]]
            embedded_features = kb.tile(embedded_features, tiling_shape)
            embedded_features = kb.bias_add(
                embedded_features, self.dimensions_bias, data_format="channels_last")
        sigma = kb.sigmoid(embedded_features)

        result = weighted_sum(first_, second_, sigma,
                              self.first_threshold, self.second_threshold)
        probs = kb.softmax(result)
        if self.return_logits:
            return [probs, result]
        return probs

    def compute_output_shape(self, input_shape):
        first_shape = input_shape[0]
        if self.return_logits:
            return [first_shape, first_shape]
        return first_shape


def TemporalDropout(inputs, dropout=0.0):
    """
    Drops with :dropout probability temporal steps of input 3D tensor
    """
    # TO DO: adapt for >3D tensors
    if dropout == 0.0:
        return inputs
    inputs_func = lambda x: kb.ones_like(inputs[:, :, 0:1])
    inputs_mask = kl.Lambda(inputs_func)(inputs)
    inputs_mask = kl.Dropout(dropout)(inputs_mask)
    tiling_shape = [1, 1, kb.shape(inputs)[2]] + [1] * (kb.ndim(inputs) - 3)
    inputs_mask = kl.Lambda(kb.tile, arguments={"n": tiling_shape},
                            output_shape=inputs._keras_shape[1:])(inputs_mask)
    answer = kl.Multiply()([inputs, inputs_mask])
    return answer


def positions_func(inputs, pad=0):
    """
    A layer filling i-th column of a 2D tensor with
    1+ln(1+i) when it contaings a meaningful symbol
    and with 0 when it contains PAD
    """
    position_inputs = kb.cumsum(kb.ones_like(inputs, dtype="float32"), axis=1)
    position_inputs *= kb.cast(kb.not_equal(inputs, pad), "float32")
    return kb.log(1.0 + position_inputs)