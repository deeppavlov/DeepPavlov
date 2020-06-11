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
from typing import List, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.matching_models.dam_utils import layers
from deeppavlov.models.ranking.matching_models.dam_utils import operations as op
from deeppavlov.models.ranking.tf_base_matching_model import TensorflowBaseMatchingModel

log = getLogger(__name__)


@register('dam_nn_use_transformer')
class DAMNetworkUSETransformer(TensorflowBaseMatchingModel):
    """
    Tensorflow implementation of Deep Attention Matching Network (DAM) [1] improved with USE [2]. We called it DAM-USE-T
    ```
    http://aclweb.org/anthology/P18-1103

    Based on Tensorflow code: https://github.com/baidu/Dialogue/tree/master/DAM
    We added USE-T [2] as a sentence encoder to the DAM network to achieve state-of-the-art performance on the datasets:
    * Ubuntu Dialogue Corpus v1 (R@1: 0.7929, R@2: 0.8912, R@5: 0.9742)
    * Ubuntu Dialogue Corpus v2 (R@1: 0.7414, R@2: 0.8656, R@5: 0.9731)

    References:
    [1]
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
    [2] Cer D, Yang Y, Kong S-y, Hua N, Limtiaco N, John RS, et al. 2018. Universal sentence encoder.
    arXiv preprint arXiv:1803.11175 2018.

    Args:
        num_context_turns (int): A number of ``context`` turns in data samples.
        max_sequence_length(int): A maximum length of text sequences in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        learning_rate (float): Initial learning rate.
        emb_matrix (np.ndarray): An embeddings matrix to initialize an embeddings layer of a model.
        trainable_embeddings (bool): Whether train embeddings matrix or not.
        embedding_dim (int): Dimensionality of token (word) embeddings.
        is_positional (bool): Adds a bunch of sinusoids of different frequencies to an embeddings.
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
        self.learning_rate = learning_rate
        self.emb_matrix = emb_matrix
        self.decay_steps = decay_steps

        super(DAMNetworkUSETransformer, self).__init__(*args, **kwargs)

        ##############################################################################
        self._init_graph()
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        ##############################################################################

        if self.load_path is not None:
            self.load()

    def _init_placeholders(self):
        """ Init model placeholders """
        with tf.variable_scope('inputs'):
            # Utterances and their lengths
            self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.num_context_turns, self.max_sentence_len))
            self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.num_context_turns))

            # Responses and their lengths
            self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
            self.response_len_ph = tf.placeholder(tf.int32, shape=(None,))

            # Labels
            self.y_true = tf.placeholder(tf.int32, shape=(None,))

            # Raw sentences for context and response
            self.context_sent_ph = tf.placeholder(tf.string,
                                                  shape=(None, self.num_context_turns),
                                                  name="context_sentences")
            self.response_sent_ph = tf.placeholder(tf.string, shape=(None,), name="response_sentences")

    def _init_sentence_encoder(self):
        """ Init sentence encoder, for example USE-T """
        # sentence encoder
        self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3",
                                trainable=False)

        # embed sentences of context
        with tf.variable_scope('sentence_embeddings'):
            x = []
            for i in range(self.num_context_turns):
                x.append(self.embed(tf.reshape(self.context_sent_ph[:, i], shape=(tf.shape(self.context_sent_ph)[0],))))
            embed_context_turns = tf.stack(x, axis=1)
            embed_response = self.embed(self.response_sent_ph)

            # for context sentences: shape=(None, self.num_context_turns, 1, 512)
            self.sent_embedder_context = tf.expand_dims(embed_context_turns, axis=2)
            # for resp sentences: shape=(None, 1, 512)
            self.sent_embedder_response = tf.expand_dims(embed_response, axis=1)

    def _init_graph(self):
        self._init_placeholders()
        self._init_sentence_encoder()

        with tf.variable_scope('sentence_emb_dim_reduction'):
            dense_emb = tf.layers.Dense(200,
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42),
                                        kernel_regularizer=tf.keras.regularizers.l2(),
                                        bias_regularizer=tf.keras.regularizers.l2(),
                                        trainable=True)

            a = []
            for i in range(self.num_context_turns):
                a.append(dense_emb(self.sent_embedder_context[:, i]))
            sent_embedder_context = tf.stack(a, axis=1)
            sent_embedder_response = dense_emb(self.sent_embedder_response)

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

        with tf.variable_scope('expand_resp_embeddings'):
            Hr = tf.concat([sent_embedder_response, Hr], axis=1)

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
        list_turn_t_sent = tf.unstack(sent_embedder_context, axis=1)

        sim_turns = []
        # for every turn_t calculate matching vector
        for turn_t, t_turn_length, turn_t_sent in zip(list_turn_t, list_turn_length, list_turn_t_sent):
            Hu = tf.nn.embedding_lookup(word_embeddings, turn_t)  # [batch, max_turn_len, emb_size]

            if self.is_positional and self.stack_num > 0:
                with tf.variable_scope('positional', reuse=True):
                    Hu = op.positional_encoding_vector(Hu, max_timescale=10)

            with tf.variable_scope('expand_cont_embeddings'):
                Hu = tf.concat([turn_t_sent, Hu], axis=1)

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
            final_info = layers.CNN_3d(sim, 32, 32)  # We can improve performance if use 32 filters for each layer
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

    def _append_sample_to_batch_buffer(self, sample: List[np.ndarray], buf: List[Tuple]):
        """
        The function for adding samples to the batch buffer

        Args:
            sample (List[nd.array]): samples generator
            buf (List[Tuple[np.ndarray]]) : List of samples with model inputs each:
                [( context, context_len, response, response_len ), ( ... ), ... ].

        Returns:
             None
        """
        sample_len = len(sample)

        batch_buffer_context = []  # [batch_size, 10, 50]
        batch_buffer_context_len = []  # [batch_size, 10]
        batch_buffer_response = []  # [batch_size, 50]
        batch_buffer_response_len = []  # [batch_size]

        raw_batch_buffer_context = []  # [batch_size, 10]
        raw_batch_buffer_response = []  # [batch_size]

        context_sentences = sample[:self.num_context_turns]
        response_sentences = sample[self.num_context_turns:sample_len // 2]

        raw_context_sentences = sample[sample_len // 2:sample_len // 2 + self.num_context_turns]
        raw_response_sentences = sample[sample_len // 2 + self.num_context_turns:]

        # Format model inputs:
        # 4 model inputs

        # 1. Token indices for context
        batch_buffer_context += [context_sentences for sent in response_sentences]  # replicate context N times
        # 2. Token indices for response
        batch_buffer_response += [response_sentence for response_sentence in response_sentences]
        # 3. Lengths of all context sentences
        lens = []
        for context in [context_sentences for sent in response_sentences]:  # replicate context N times
            context_sentences_lens = []
            for sent in context:
                sent_len = len(sent[sent != 0])
                sent_len = sent_len + 1 if sent_len > 0 else 0  # 1 additional token is the USE token
                context_sentences_lens.append(sent_len)
            lens.append(context_sentences_lens)
        batch_buffer_context_len += lens
        # 4. Length of response
        lens = []
        for response in [response_sentence for response_sentence in response_sentences]:
            sent_len = len(response[response != 0])
            sent_len = sent_len + 1 if sent_len > 0 else 0  # 1 additional token is the USE token
            lens.append(sent_len)
        batch_buffer_response_len += lens
        # 5. Raw context sentences
        raw_batch_buffer_context += [raw_context_sentences for sent in raw_response_sentences]
        # 6. Raw response sentences
        raw_batch_buffer_response += [raw_sent for raw_sent in raw_response_sentences]

        for i in range(len(batch_buffer_context)):
            buf.append(tuple((
                batch_buffer_context[i],
                batch_buffer_context_len[i],
                batch_buffer_response[i],
                batch_buffer_response_len[i],
                raw_batch_buffer_context[i],
                raw_batch_buffer_response[i]
            )))
        return len(response_sentences)

    def _make_batch(self, batch: List[Tuple[np.ndarray]]) -> Dict:
        """
        The function for formatting model inputs

        Args:
            batch (List[Tuple[np.ndarray]]): List of samples with model inputs each:
                [( context, context_len, response, response_len ), ( ... ), ... ].
            graph (str): which graph the inputs is preparing for

        Returns:
            Dict: feed_dict to feed a model
        """
        input_context = []
        input_context_len = []
        input_response = []
        input_response_len = []
        input_raw_context = []
        input_raw_response = []

        # format model inputs for MAIN graph as numpy arrays
        for sample in batch:
            input_context.append(sample[0])
            input_context_len.append(sample[1])
            input_response.append(sample[2])
            input_response_len.append(sample[3])
            input_raw_context.append(sample[4])  # raw context is the 4th element of each Tuple in the batch
            input_raw_response.append(sample[5])  # raw response is the 5th element of each Tuple in the batch

        return {
            self.utterance_ph: np.array(input_context),
            self.all_utterance_len_ph: np.array(input_context_len),
            self.response_ph: np.array(input_response),
            self.response_len_ph: np.array(input_response_len),
            self.context_sent_ph: np.array(input_raw_context),
            self.response_sent_ph: np.array(input_raw_response)
        }
