# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
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
from typing import Optional

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.matching_models.dam_utils import layers
from deeppavlov.models.ranking.matching_models.dam_utils import operations as op
from deeppavlov.models.ranking.tf_base_matching_model import TensorflowBaseMatchingModel

log = getLogger(__name__)


@register('dam_nn')
class DAMNetwork(TensorflowBaseMatchingModel):
    """
    Tensorflow implementation of Deep Attention Matching Network (DAM)

    ```
    @inproceedings{ ,
      title={Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network},
      author={Xiangyang Zhou, Lu Li, Daxiang Dong, Yi Liu, Ying Chen, Wayne Xin Zhao, Dianhai Yu and Hua Wu},
      booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      volume={1},
      pages={  --  },
      year={2018}
    }
    ```
    http://aclweb.org/anthology/P18-1103

    Based on authors' Tensorflow code: https://github.com/baidu/Dialogue/tree/master/DAM

    Args:
        num_context_turns (int): A number of ``context`` turns in data samples.
        max_sequence_length(int): A maximum length of text sequences in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        learning_rate (float): Initial learning rate.
        emb_matrix (np.ndarray): An embeddings matrix to initialize an embeddings layer of a model.
        trainable_embeddings (bool): Whether train embeddings matrix or not.
        embedding_dim (int): Dimensionality of token (word) embeddings.
        is_positional (bool): Adds a bunch of sinusoids of different frequencies to an embeddings.
        filters2_conv3d (int): number of filters in the second conv3d layer (cnn aggregation). Default: 16.
        stack_num (int): Number of stack layers, default is 5.
        seed (int): Random seed.
        decay_steps (int): Number of steps after which is to decay the learning rate.
    """

    def __init__(self,
                 embedding_dim: int = 200,
                 max_sequence_length: int = 50,
                 learning_rate: float = 1e-3,
                 emb_matrix: Optional[np.ndarray] = None,
                 trainable_embeddings: bool = False,
                 is_positional: bool = True,
                 filters2_conv3d: int = 16,
                 stack_num: int = 5,
                 seed: int = 65,
                 decay_steps: int = 600,
                 *args,
                 **kwargs):

        self.seed = seed
        tf.set_random_seed(self.seed)

        self.max_sentence_len = max_sequence_length
        self.word_embedding_size = embedding_dim
        self.trainable = trainable_embeddings
        self.is_positional = is_positional
        self.stack_num = stack_num
        self.filters2_conv3d = filters2_conv3d

        self.learning_rate = learning_rate
        self.emb_matrix = emb_matrix
        self.decay_steps = decay_steps

        super(DAMNetwork, self).__init__(*args, **kwargs)

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)
        self._init_graph()
        self.sess.run(tf.global_variables_initializer())

        if self.load_path is not None:
            self.load()

    def _init_placeholders(self):
        with tf.variable_scope('inputs'):
            # Utterances and their lengths
            self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.num_context_turns, self.max_sentence_len))
            self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.num_context_turns))

            # Responses and their lengths
            self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
            self.response_len_ph = tf.placeholder(tf.int32, shape=(None,))

            # Labels
            self.y_true = tf.placeholder(tf.int32, shape=(None,))

    def _init_graph(self):
        self._init_placeholders()

        with tf.variable_scope('embedding_matrix_init'):
            word_embeddings = tf.get_variable("word_embeddings_v",
                                              initializer=tf.constant(self.emb_matrix, dtype=tf.float32),
                                              trainable=self.trainable)
        with tf.variable_scope('embedding_lookup'):
            response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)

        Hr = response_embeddings
        if self.is_positional and self.stack_num > 0:
            with tf.variable_scope('positional'):
                Hr = op.positional_encoding_vector(Hr, max_timescale=10)

        Hr_stack = [Hr]

        for index in range(self.stack_num):
            with tf.variable_scope('self_stack_' + str(index)):
                Hr = layers.block(
                    Hr, Hr, Hr,
                    Q_lengths=self.response_len_ph, K_lengths=self.response_len_ph, attention_type='dot')
                Hr_stack.append(Hr)

        # context part
        # a list of length max_turn_num, every element is a tensor with shape [batch, max_turn_len]
        list_turn_t = tf.unstack(self.utterance_ph, axis=1)
        list_turn_length = tf.unstack(self.all_utterance_len_ph, axis=1)

        sim_turns = []
        # for every turn_t calculate matching vector
        for turn_t, t_turn_length in zip(list_turn_t, list_turn_length):
            Hu = tf.nn.embedding_lookup(word_embeddings, turn_t)  # [batch, max_turn_len, emb_size]

            if self.is_positional and self.stack_num > 0:
                with tf.variable_scope('positional', reuse=True):
                    Hu = op.positional_encoding_vector(Hu, max_timescale=10)
            Hu_stack = [Hu]

            for index in range(self.stack_num):
                with tf.variable_scope('self_stack_' + str(index), reuse=True):
                    Hu = layers.block(
                        Hu, Hu, Hu,
                        Q_lengths=t_turn_length, K_lengths=t_turn_length, attention_type='dot')

                    Hu_stack.append(Hu)

            r_a_t_stack = []
            t_a_r_stack = []
            for index in range(self.stack_num + 1):

                with tf.variable_scope('t_attend_r_' + str(index)):
                    try:
                        t_a_r = layers.block(
                            Hu_stack[index], Hr_stack[index], Hr_stack[index],
                            Q_lengths=t_turn_length, K_lengths=self.response_len_ph, attention_type='dot')
                    except ValueError:
                        tf.get_variable_scope().reuse_variables()
                        t_a_r = layers.block(
                            Hu_stack[index], Hr_stack[index], Hr_stack[index],
                            Q_lengths=t_turn_length, K_lengths=self.response_len_ph, attention_type='dot')

                with tf.variable_scope('r_attend_t_' + str(index)):
                    try:
                        r_a_t = layers.block(
                            Hr_stack[index], Hu_stack[index], Hu_stack[index],
                            Q_lengths=self.response_len_ph, K_lengths=t_turn_length, attention_type='dot')
                    except ValueError:
                        tf.get_variable_scope().reuse_variables()
                        r_a_t = layers.block(
                            Hr_stack[index], Hu_stack[index], Hu_stack[index],
                            Q_lengths=self.response_len_ph, K_lengths=t_turn_length, attention_type='dot')

                t_a_r_stack.append(t_a_r)
                r_a_t_stack.append(r_a_t)

            t_a_r_stack.extend(Hu_stack)
            r_a_t_stack.extend(Hr_stack)

            t_a_r = tf.stack(t_a_r_stack, axis=-1)
            r_a_t = tf.stack(r_a_t_stack, axis=-1)

            # log.info(t_a_r, r_a_t)  # debug

            # calculate similarity matrix
            with tf.variable_scope('similarity'):
                # sim shape [batch, max_turn_len, max_turn_len, 2*stack_num+1]
                # divide sqrt(200) to prevent gradient explosion
                sim = tf.einsum('biks,bjks->bijs', t_a_r, r_a_t) / tf.sqrt(float(self.word_embedding_size))

            sim_turns.append(sim)

        # cnn and aggregation
        sim = tf.stack(sim_turns, axis=1)
        log.info('sim shape: %s' % sim.shape)
        with tf.variable_scope('cnn_aggregation'):
            final_info = layers.CNN_3d(sim, 32, self.filters2_conv3d)
            # for douban
            # final_info = layers.CNN_3d(sim, 16, 16)

        # loss and train
        with tf.variable_scope('loss'):
            self.loss, self.logits = layers.loss(final_info, self.y_true, clip_value=10.)
            self.y_pred = tf.nn.softmax(self.logits, name="y_pred")
            tf.summary.scalar('loss', self.loss)

            self.global_step = tf.Variable(0, trainable=False)
            initial_learning_rate = self.learning_rate
            self.learning_rate = tf.train.exponential_decay(
                initial_learning_rate,
                global_step=self.global_step,
                decay_steps=self.decay_steps,
                decay_rate=0.9,
                staircase=True)

            Optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.grads_and_vars = Optimizer.compute_gradients(self.loss)

            for grad, var in self.grads_and_vars:
                if grad is None:
                    log.info(var)

            self.capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.grads_and_vars]
            self.train_op = Optimizer.apply_gradients(
                self.capped_gvs,
                global_step=self.global_step)

        # Debug
        self.print_number_of_parameters()
