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

from typing import Optional, Union

import tensorflow as tf

from deeppavlov.core.layers.embeddings import AdvancedEmbedding
from deeppavlov.core.layers.attention import MultiHeadSelfAttention
from deeppavlov.core.layers.activations import gelu

# LayerNormalization is shipped in TF since v1.14, so in case of previous versions we use a custom implementation
try:
    from tensorflow.keras.layers import LayerNormalization
except ImportError:
    from deeppavlov.core.layers.normalization import LayerNormalization


class BERT(tf.keras.Model):
    """
    BERT body (could be also implemented as a tensorflow.python.keras.engine.network.Network subclass in order to have
    just weight-(de)serialization methods). Higher level models (e.g. classifiers or taggers) could be implemented as a
    child of this class, rather than incorporate it as a building block.
    All naming of sublayers with trainable variables is performed in order for users to be able to load official
    checkpoints from Google. All the default parameters for this layer and all sublayers are in accordance with
    multilingual BERT Base.

    Args:
        return_stack: by default (None) return pooled output; if False, return a sequence from the last encoder layer;
            if True, return collection of sequence outputs from all encoder layers
        vocab_size: size of the token embedding vocabulary
        token_type_vocab_size: the vocabulary size of `token_type_ids` (or `segment_ids`)
        sep_token_index: index of separator-token used for calculating `token_type_ids` (or `segment_ids`)
        pad_token_index: index of the padding token used for mask computation
        emb_dropout_rate: probability of dropping out layer-normalized embeddings
        use_one_hot_embeddings: if True, use one-hot method for word embeddings; if False, use
            `tf.nn.embedding_lookup()`. One hot is better for TPUs.
        max_len: maximum sequence length that might ever be used with this model. This can be longer than the sequence
            length of input_tensor, but cannot be shorter.
        initializer_range: stddev for embedding initialization range (TruncatedNormal initializer is used)
        trainable_pos_embedding: whether the positional embedding matrix is trainable
        layer_norm_epsilon: some small number to avoid zero division during layer normalization
        hidden_size: hidden size of the Transformer (d_model)
        intermediate_size: the size of the intermediate dense layer
        num_hidden_layers: number of layers (blocks) in the Transformer
        num_heads: number of attention heads in the Transformer
        hidden_dropout_prob: dropout probability for the hidden layers
        attention_probs_dropout_prob: dropout probability of the attention probabilities
        intermediate_act_fn: the non-linear activation function to apply to the output of the intermediate dense layer
        pooler_fc_size: the size of the dense layer on top of the first step of encoder output
        trainable: whether the layer's variables should be trainable
        **kwargs: keyword arguments for base Layer class
    """
    def __init__(self,
                 return_stack: Optional[bool] = None,
                 vocab_size: int = 119547,
                 token_type_vocab_size: int = 2,
                 sep_token_index: int = 102,
                 pad_token_index: int = 0,
                 emb_dropout_rate: float = 0.1,  # equal to hidden_dropout_prob in the original implementation
                 use_one_hot_embedding: bool = False,
                 max_len: int = 512,
                 initializer_range: float = 0.02,
                 trainable_pos_embedding: bool = True,
                 layer_norm_epsilon: float = 1e-12,
                 # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/layers/python/layers/layers.py#L2315
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_hidden_layers: int = 12,
                 num_heads: int = 12,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 pooler_fc_size: int = 768,
                 trainable: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.supports_masking = True

        self.return_stack = return_stack
        self.pad_token_index = pad_token_index

        # use name scopes for compatibility with both eager and graph modes
        with tf.name_scope('embeddings'):
            self.embed = AdvancedEmbedding(vocab_size=vocab_size,
                                           token_type_vocab_size=token_type_vocab_size,
                                           sep_token_index=sep_token_index,
                                           output_dim=hidden_size,
                                           use_one_hot_embedding=use_one_hot_embedding,
                                           max_len=max_len,
                                           initializer_range=initializer_range,
                                           trainable_pos_embedding=trainable_pos_embedding,
                                           trainable=trainable,
                                           name='embeddings')
            self.embed_dropout = tf.keras.layers.Dropout(rate=emb_dropout_rate,
                                                         trainable=trainable,
                                                         name='embeddings/dropout')
            self.embed_layer_norm = LayerNormalization(epsilon=layer_norm_epsilon,
                                                       trainable=trainable,
                                                       name='embeddings/LayerNorm')

        with tf.name_scope('encoder'):
            self.encoder = tf.keras.Sequential(name='encoder')
            for i in range(num_hidden_layers):
                with tf.name_scope(f'layer_{i}'):
                    self.encoder.add(TransformerBlock(hidden_size=hidden_size,
                                                      intermediate_size=intermediate_size,
                                                      num_heads=num_heads,
                                                      hidden_dropout_prob=hidden_dropout_prob,
                                                      attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                      intermediate_act_fn=intermediate_act_fn,
                                                      layer_norm_epsilon=layer_norm_epsilon,
                                                      trainable=trainable,
                                                      name=f'layer_{i}'))
        # The "pooler" converts the encoded sequence tensor of shape [batch_size, seq_length, hidden_size] to a tensor
        # of shape [batch_size, hidden_size]. This is necessary for segment-level (or segment-pair-level) classification
        # tasks where we need a fixed dimensional representation of the segment.
        # We "pool" the model by simply taking the hidden state corresponding to the first token. We assume that this
        # has been pre-trained
        self.pooler = tf.keras.layers.Dense(pooler_fc_size, activation='tanh', trainable=trainable, name='pooler/dense')

    def call(self,
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:

        emb = self.embed(inputs, training=training)
        emb_norm_do = self.embed_dropout(self.embed_layer_norm(emb), training=training)

        # always compute mask if it is not provided
        if mask is None:
            mask = tf.cast(tf.not_equal(inputs, self.pad_token_index), tf.int32)
        enc = self.encoder(emb_norm_do, training=training, mask=mask)
        # For classification tasks, the first vector (corresponding to [CLS]) is used as the "sentence vector". Note
        # that this only makes sense because the entire model is fine-tuned.
        po = self.pooler(tf.squeeze(enc[:, 0:1, :], axis=1))

        if self.return_stack is None:
            return po
        elif self.return_stack:
            raise NotImplementedError('Currently all encoder layers output could not be obtained. You can get '
                                      'sequence output from the last encoder layer setting return_stack to False')
        else:
            return enc


class TransformerBlock(tf.keras.layers.Layer):
    """
    One block of transformer architecture.

    Args:
        hidden_size: hidden size of the Transformer (d_model)
        intermediate_size: the size of the intermediate dense layer
        num_heads: number of attention heads in the Transformer
        hidden_dropout_prob: dropout probability for the hidden layers
        attention_probs_dropout_prob: dropout probability of the attention probabilities
        intermediate_act_fn: the non-linear activation function to apply to the output of the intermediate dense layer
        layer_norm_epsilon: some small number to avoid zero division during layer normalization
        trainable: whether the layer's variables should be trainable
        **kwargs: keyword arguments for base Layer class
    """
    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 num_heads: int = 12,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 intermediate_act_fn: Union[str, callable] = gelu,
                 layer_norm_epsilon: float = 1e-12,
                 trainable: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.supports_masking = True

        self.mhsa = MultiHeadSelfAttention(hidden_size=hidden_size,
                                           num_heads=num_heads,
                                           attention_probs_dropout_prob=attention_probs_dropout_prob,
                                           trainable=trainable,
                                           name='attention')
        # Run a linear projection of `hidden_size` then add a residual with `layer_input`.
        self.dense = tf.keras.layers.Dense(units=hidden_size, trainable=trainable, name='attention/output/dense')

        self.dropout1 = tf.keras.layers.Dropout(rate=hidden_dropout_prob, name='attention/output/dropout')
        self.layernorm1 = LayerNormalization(epsilon=layer_norm_epsilon,
                                             trainable=trainable,
                                             name='attention/output/LayerNorm')

        # Point-wise Feed Forward network
        # The activation is only applied to the "intermediate" hidden layer.
        self.pff1 = tf.keras.layers.Dense(units=intermediate_size,
                                          activation=intermediate_act_fn,
                                          trainable=trainable,
                                          name='intermediate/dense')
        # Down-project back to `hidden_size` then add the residual.
        self.pff2 = tf.keras.layers.Dense(units=hidden_size, trainable=trainable, name='output/dense')

        self.dropout2 = tf.keras.layers.Dropout(rate=hidden_dropout_prob)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_epsilon, trainable=trainable, name='output/LayerNorm')

    def call(self,
             inputs: tf.Tensor,
             training: Optional[bool] = None,
             mask: Optional[bool] = None,
             **kwargs) -> tf.Tensor:

        attn_output = self.mhsa(inputs, mask=mask)
        attn_output = self.dropout1(self.dense(attn_output), training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.pff2(self.pff1(out1))
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        # workaround for mask propagation
        out2._keras_mask = mask  # TODO: try to get rid of this. This even could not be moved to self-attention layer.
        return out2

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape
