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

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops


class KBAttention(base.Layer):
    # TODO: update class doc
    """Densely-connected layer class.
    Arguments:
        units: Integer or Long, dimensionality of the output space.
        activation: Activation function (callable). Set it to None to maintain a
          linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: Initializer function for the weight matrix.
          If ``None`` (default), weights are initialized using the default
          initializer used by `tf.get_variable`.
        bias_initializer: Initializer function for the bias.
        kernel_regularizer: Regularizer function for the weight matrix.
        bias_regularizer: Regularizer function for the bias.
        activity_regularizer: Regularizer function for the output.
        kernel_constraint: An optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        bias_constraint: An optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` also add variables to the graph collection
          `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: String, the name of the layer. Layers with the same name will
          share weights, but to avoid mistakes we require reuse=True in such cases.
        reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.
    Properties:
        units: Python integer, dimensionality of the output space.
        activation: Activation function (callable).
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: Initializer instance (or name) for the kernel matrix.
        bias_initializer: Initializer instance (or name) for the bias.
        kernel_regularizer: Regularizer instance for the kernel matrix (callable)
        bias_regularizer: Regularizer instance for the bias (callable).
        activity_regularizer: Regularizer instance for the output (callable)
        kernel_constraint: Constraint function for the kernel matrix.
        bias_constraint: Constraint function for the bias.
        kernel: Weight matrix (TensorFlow variable or tensor).
        bias: Bias vector, if applicable (TensorFlow variable or tensor).
    """

    def __init__(self, units, hidden_sizes,
                 kb_inputs,
                 kb_mask,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 reuse=None,
                 **kwargs):
        super(KBAttention, self).__init__(trainable=trainable, name=name,
                                          activity_regularizer=activity_regularizer,
                                          *kwargs)
        self.units = units
        self.hidden_sizes = hidden_sizes
        self.kb_inputs = kb_inputs
        self.kb_mask = kb_mask
        self.kb_input_shape = kb_inputs.get_shape().as_list()
        self.dense_name = name or "mlp"
        self.dense_params = {
            "activation": activation,
            "use_bias": use_bias,
            "kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer,
            "kernel_regularizer": kernel_regularizer,
            "bias_regularizer": bias_regularizer,
            "activity_regularizer": activity_regularizer,
            "kernel_constraint": kernel_constraint,
            "bias_constraint": bias_constraint,
            "trainable": trainable,
            "dtype": self.kb_inputs.dtype.base_dtype,
            "_reuse": reuse
        }
        # print("KB shape =", self.kb_input_shape)

    def build(self, input_shape):
        # if in_shape[:-1] != self.kb_inputs.shape 
        # TODO: check input shape
        # print("in build")
        in_shape = input_shape[:1].concatenate(self.kb_input_shape)
        in_shape = in_shape[:-1].concatenate(in_shape[-1] + input_shape[-1])
        # print("first in_shape =", in_shape)
        self.layers = []
        for i, size in enumerate(self.hidden_sizes):
            name = self.dense_name
            if name is not None:
                name = name + '{:d}'.format(i)
            layer = tf.layers.Dense(size, name=name, _scope=name, **self.dense_params)
            layer.build(in_shape)
            in_shape = layer.compute_output_shape(in_shape)

            self.layers.append(layer)

        # print("input_shape =", input_shape)
        # print("last in_shape =", in_shape)
        # in_shape = in_shape[:-2].concatenate(in_shape[-2] + input_shape[-1])
        # print("last in_shape =", in_shape)
        self.output_layer = tf.layers.Dense(self.units, **self.dense_params)
        self.output_layer.build(input_shape)
        # print("build = True")
        self.built = True

    def call(self, inputs):
        # print("in call")
        # TODO: check input dtype

        # Tile kb_inputs
        kb_inputs = self.kb_inputs
        for i in range(inputs.shape.ndims - 1):
            kb_inputs = tf.expand_dims(kb_inputs, 0)
        kb_inputs = tf.tile(kb_inputs, tf.concat((tf.shape(inputs)[:-1], [1, 1]), 0))

        # Expand kb_mask
        kb_mask = self.kb_mask
        for i in range(inputs.shape.ndims - 2):
            kb_mask = tf.expand_dims(kb_mask, 1)
        kb_mask = tf.expand_dims(kb_mask, -1)

        # Tile inputs
        kb_size = tf.shape(self.kb_inputs)[0]
        tiling = tf.concat(([1] * (inputs.shape.ndims - 1), [kb_size], [1]), 0)
        cell_inputs = tf.tile(tf.expand_dims(inputs, -2), tiling)

        outputs = tf.concat([kb_inputs, cell_inputs], -1)
        outputs = tf.multiply(outputs, kb_mask)
        for layer in self.layers:
            outputs = layer.call(outputs)
        # outputs = tf.Print(outputs, [outputs], "KB attention pre-last layer output =")
        outputs = tf.squeeze(outputs, [-1])
        # print("inputs shape =", inputs.shape)
        # print("outputs shape =", outputs.shape)
        outputs = tf.concat([self.output_layer(inputs), outputs], -1)
        # print("out of call")
        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        output_shape = input_shape[:-1].concatenate(self.units + self.kb_input_shape[0])
        # print("computed output shape is", output_shape)
        return output_shape

    def compute_output_shape(self, input_shape):
        return self._compute_output_shape(input_shape)
