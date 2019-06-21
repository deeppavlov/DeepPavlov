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

from typing import Optional

import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):
    """
    Layer that normalize outputs to statistics aggregated across hidden dimension.

    Args:
        epsilon: some small number to avoid zero division
        **kwargs: keyword arguments for base Layer class
    """
    def __init__(self,
                 epsilon=1e-12,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.supports_masking = True

        self.eps = tf.constant(epsilon, dtype=tf.float32)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer=tf.keras.initializers.Zeros())
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Ones())
        super().build(input_shape)

    def call(self,
             x: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> None:
        u = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        s = tf.math.reduce_mean(tf.math.square(x - u), axis=-1, keepdims=True)
        z = (x - u) / tf.math.sqrt(s + self.eps)
        return self.gamma * z + self.beta

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape
