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

from typing import List, Iterable, Dict, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.tf_base_matching_model import TensorflowBaseMatchingModel
from deeppavlov.models.ranking.matching_models.dam_utils import layers
from deeppavlov.models.ranking.matching_models.dam_utils import operations as op

log = get_logger(__name__)


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
                 batch_size: int,
                 embedding_dim: int = 200,
                 num_context_turns: int = 10,
                 max_sequence_length: int = 50,
                 learning_rate: float = 1e-3,
                 emb_matrix: np.ndarray = None,
                 trainable_embeddings: bool = False,
                 is_positional: bool = True,
                 stack_num: int = 5,
                 seed: int = 65,
                 decay_steps: int = 600,
                 *args,
                 **kwargs):

        self.seed = seed
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.batch_size = batch_size
        self.num_context_turns = num_context_turns
        self.max_sentence_len = max_sequence_length
        self.word_embedding_size = embedding_dim
        self.trainable = trainable_embeddings
        self.is_positional = is_positional
        self.stack_num = stack_num
        self.learning_rate = learning_rate
        self.emb_matrix = emb_matrix
        self.decay_steps = decay_steps

        ##############################################################################
        self.g_use = tf.Graph()
        with self.g_use.as_default():
            # sentence encoder
            self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3", trainable=False)

            # Raw sentences for context and response
            self.context_sent_ph = tf.placeholder(tf.string,
                                                  shape=(None, self.num_context_turns),
                                                  name="context_sentences")
            self.response_sent_ph = tf.placeholder(tf.string, shape=(None,), name="response_sentences")
            # embed sentences of context
            with tf.variable_scope('sentence_embeddings'):
                x = []
                for i in range(self.num_context_turns):
                    x.append(self.embed(tf.reshape(self.context_sent_ph[:, i], shape=(tf.shape(self.context_sent_ph)[0], ))))
                embed_context_turns = tf.stack(x, axis=1)
                embed_response = self.embed(self.response_sent_ph)

                self.sent_embedder_context = tf.expand_dims(embed_context_turns, axis=2)
                self.sent_embedder_response = tf.expand_dims(embed_response, axis=1)

        self.cpu_sess = tf.Session(config=tf.ConfigProto(), graph=self.g_use)
        with self.g_use.as_default():
            self.cpu_sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        ##############################################################################

        ##############################################################################
        self._init_graph()

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)
        self.sess.run(tf.global_variables_initializer())
        ##############################################################################

        super(DAMNetworkUSETransformer, self).__init__(
            batch_size=batch_size, num_context_turns=num_context_turns, *args, **kwargs)

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

            # Raw sentences for context and response
            self.context_sent_emb_ph = tf.placeholder(tf.float32, shape=(None, self.num_context_turns, 1, 512))
            self.response_sent_emb_ph = tf.placeholder(tf.float32, shape=(None, 1, 512))

    def _init_graph(self):
        self._init_placeholders()

        with tf.variable_scope('sentence_emb_dim_reduction'):
            dense_emb = tf.layers.Dense(200,
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=42),
                                        kernel_regularizer=tf.keras.regularizers.l2(),
                                        bias_regularizer=tf.keras.regularizers.l2(),
                                        trainable=True)

            a = []
            for i in range(self.num_context_turns):
                a.append(dense_emb(self.context_sent_emb_ph[:, i]))
            sent_embedder_context = tf.stack(a, axis=1)
            sent_embedder_response = dense_emb(self.response_sent_emb_ph)

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
        list_turn_t      = tf.unstack(self.utterance_ph, axis=1)
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
            final_info = layers.CNN_3d(sim, 32, 32)    # We can improve performance if use 32 filters for each layer
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

        batch_buffer_context = []       # [batch_size, 10, 50]
        batch_buffer_context_len = []   # [batch_size, 10]
        batch_buffer_response = []      # [batch_size, 50]
        batch_buffer_response_len = []  # [batch_size]

        raw_batch_buffer_context = []   # [batch_size, 10]
        raw_batch_buffer_response = []  # [batch_size]

        context_sentences = sample[:self.num_context_turns]
        response_sentences = sample[self.num_context_turns:sample_len // 2]

        raw_context_sentences = sample[sample_len // 2:sample_len // 2 + self.num_context_turns]
        raw_response_sentences = sample[sample_len // 2 + self.num_context_turns:]

        # Format model inputs:
        # 4 model inputs

        # 1. Token indices for context
        batch_buffer_context += [context_sentences for sent in response_sentences]
        # 2. Token indices for response
        batch_buffer_response += [response_sentence for response_sentence in response_sentences]
        # 3. Lens of context sentences
        lens = []
        for context in [context_sentences for sent in response_sentences]:
            context_sentences_lens = []
            for sent in context:
                context_sentences_lens.append(len(sent[sent != 0]))
            lens.append(context_sentences_lens)
        batch_buffer_context_len += lens
        # 4. Lens of context sentences
        lens = []
        for context in [response_sentence for response_sentence in response_sentences]:
            lens.append(len(context[context != 0]))
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

    def _make_batch(self, batch: List[Tuple[np.ndarray]], graph: str = "main") -> Dict:
        """
        The function for formatting model inputs

        Args:
            batch (List[Tuple[np.ndarray]]): List of samples with model inputs each:
                [( context, context_len, response, response_len ), ( ... ), ... ].
            graph (str): which graph the inputs is preparing for

        Returns:
            Dict: feed_dict to feed a model
        """
        if graph == "use":
            input_raw_context = []
            input_raw_response = []

            # format model inputs for USE graph as numpy arrays
            for sample in batch:
                input_raw_context.append(sample[4])   # raw context is the 4th element of each Tuple in the batch
                input_raw_response.append(sample[5])  # raw response is the 5th element of each Tuple in the batch

            return {
                self.context_sent_ph: np.array(input_raw_context),
                self.response_sent_ph: np.array(input_raw_response)
            }
        elif graph == "main":
            input_context = []
            input_context_len = []
            input_response = []
            input_response_len = []

            # format model inputs for MAIN graph as numpy arrays
            for sample in batch:
                input_context.append(sample[0])
                input_context_len.append(sample[1])
                input_response.append(sample[2])
                input_response_len.append(sample[3])

            return {
                self.utterance_ph: np.array(input_context),
                self.all_utterance_len_ph: np.array(input_context_len),
                self.response_ph: np.array(input_response),
                self.response_len_ph: np.array(input_response_len)
            }

    def _predict_on_batch(self, batch: Dict, graph: str = "main") -> np.ndarray:
        """
        Run a model with the batch of inputs.
        The function returns a list of predictions for the batch in numpy format

        Args:
            batch (Dict): feed_dict that contains a batch with inputs for a model
            graph (str): which graph the inputs is preparing for

        Returns:
            nd.array: predictions for the batch
        """
        if graph == "use":
            return self.cpu_sess.run([self.sent_embedder_context, self.sent_embedder_response],
                                     feed_dict=batch)
        elif graph == "main":
            return self.sess.run(self.y_pred, feed_dict=batch)[:, 1]

    def _train_on_batch(self, batch: Dict, y: List[int]) -> float:
        """
        The function is for formatting of feed_dict used as an input for a model
        Args:
            batch (Dict): feed_dict that contains a batch with inputs for a model (except ground truth labels)
            y (List(int)): list of ground truth labels

        Returns:
            float: value of mean loss on the batch
        """

        batch.update({self.y_true: np.array(y)})
        return self.sess.run([self.loss, self.train_op], feed_dict=batch)[0]  # return the first item aka loss

    def __call__(self, samples_generator: Iterable[List[np.ndarray]]) -> np.ndarray:
        """
        This method is called by trainer to make one evaluation step on one batch.

        Args:
            samples_generator (Iterable[List[np.ndarray]]):  generator that returns list of numpy arrays
            of words of all sentences represented as integers.
            Has shape: (number_of_context_turns + 1, max_number_of_words_in_a_sentence)

        Returns:
            np.ndarray: predictions for the batch of samples
        """

        y_pred = []
        buf = []
        for j, sample in enumerate(samples_generator, start=1):
            n_responses = len(sample[self.num_context_turns:len(sample) // 2])
            self._append_sample_to_batch_buffer(sample, buf)
            if len(buf) >= self.batch_size:
                for i in range(len(buf) // self.batch_size):
                    # 1. USE Graph
                    fd = self._make_batch(buf[i * self.batch_size:(i + 1) * self.batch_size], graph="use")
                    context_emb, response_emb = self._predict_on_batch(fd, graph="use")

                    # 2. MAIN Graph
                    fd = self._make_batch(buf[i * self.batch_size:(i + 1) * self.batch_size], graph="main")
                    fd.update({
                        self.context_sent_emb_ph: context_emb,
                        self.response_sent_emb_ph: response_emb
                    })
                    yp = self._predict_on_batch(fd, graph="main")
                    y_pred += list(yp)
                lenb = len(buf) % self.batch_size
                if lenb != 0:
                    buf = buf[-lenb:]
                else:
                    buf = []
        if len(buf) != 0:
            # 1. USE Graph
            fd = self._make_batch(buf, graph="use")
            context_emb, response_emb = self._predict_on_batch(fd, graph="use")

            # 2. MAIN Graph
            fd = self._make_batch(buf, graph="main")
            fd.update({
                self.context_sent_emb_ph: context_emb,
                self.response_sent_emb_ph: response_emb
            })
            yp = self._predict_on_batch(fd, graph="main")
            y_pred += list(yp)
        y_pred = np.asarray(y_pred)
        # reshape to [batch_size, n_responses] if needed (n_responses > 1)
        y_pred = np.reshape(y_pred, (j, n_responses)) if n_responses > 1 else y_pred
        return y_pred

    def train_on_batch(self, samples_generator: Iterable[List[np.ndarray]], y: List[int]) -> float:
        """
        This method is called by trainer to make one training step on one batch.

        Args:
            samples_generator (Iterable[List[np.ndarray]]): generator that returns list of numpy arrays
            of words of all sentences represented as integers.
            Has shape: (number_of_context_turns + 1, max_number_of_words_in_a_sentence)
            y (List[int]): tuple of labels, with shape: (batch_size, )

        Returns:
            float: value of mean loss on the batch
        """
        buf = []
        for sample in samples_generator:
            self._append_sample_to_batch_buffer(sample, buf)

        fd = self._make_batch(buf, graph="use")
        context_emb, response_emb = self._predict_on_batch(fd, graph="use")   # We do not update USE weights

        # 2. MAIN Graph
        fd = self._make_batch(buf, graph="main")
        fd.update({
            self.context_sent_emb_ph: context_emb,
            self.response_sent_emb_ph: response_emb
        })
        loss = self._train_on_batch(fd, y)                                    # We do update MAIN model weights
        return loss