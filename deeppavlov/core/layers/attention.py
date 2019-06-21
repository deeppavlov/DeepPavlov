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


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Simplified version of multi-head attention. It uses just an input mask (if provided) to calculate attention mask.

    Args:
          hidden_size: hidden size of the Transformer (d_model)
          num_heads: number of attention heads
          neg_inf: some number close to -infinity to add to logits in order to mask corresponding values
          attention_probs_dropout_prob: dropout probability of the attention probabilities.
          trainable: whether the layer's variables should be trainable
          **kwargs: keyword arguments for base Layer class
    """
    def __init__(self,
                 hidden_size: int = 768,
                 num_heads: int = 12,
                 neg_inf: float = -10000.0,
                 attention_probs_dropout_prob: float = 0.1,
                 trainable: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.supports_masking = True

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.neg_inf = neg_inf

        assert hidden_size % self.num_heads == 0

        self.depth = hidden_size // self.num_heads

        self.wq = tf.keras.layers.Dense(hidden_size, trainable=trainable, name='self/query')
        self.wk = tf.keras.layers.Dense(hidden_size, trainable=trainable, name='self/key')
        self.wv = tf.keras.layers.Dense(hidden_size, trainable=trainable, name='self/value')
        self.dropout = tf.keras.layers.Dropout(rate=attention_probs_dropout_prob)

    def split_heads(self, x: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self,
             x: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[tf.Tensor] = None,
             **kwargs) -> None:

        batch_size = tf.shape(x)[0]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(q, k, transpose_b=True)
        # Scale attention_scores. The dot-product attention is scaled by a factor of square root of the depth.
        # This is done because for large values of depth, the dot product grows large in magnitude pushing the softmax
        # function where it has small gradients resulting in a very hard softmax.
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        # Square root of dk is used for scaling (and not any other number) because the matmul of Q and K should have
        # a mean of 0 and variance of 1, so that we get a gentler softmax.
        scaled_attention_logits = attention_scores / tf.math.sqrt(dk)

        # add the mask to the scaled tensor
        if mask is not None:
            attention_mask = tf.cast(mask, tf.float32)[:, tf.newaxis, tf.newaxis, :]
            scaled_attention_logits += ((1.0 - attention_mask) * self.neg_inf)

        # Normalize the attention scores to probabilities
        attention_weights = tf.nn.softmax(scaled_attention_logits)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        scaled_attention = self.dropout(tf.matmul(attention_weights, v), training=training)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # concatenate all heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.hidden_size))

        return concat_attention

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape
