import numpy as np
import os

import tensorflow as tf
from nltk import word_tokenize
from tensorflow.python.ops import rnn_cell_impl
from tqdm import tqdm

from deeppavlov.models.lstms.sru import SRUCell

rnn_cell = rnn_cell_impl
rnn = tf.contrib.rnn

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.attributes import check_attr_true

config = tf.ConfigProto(gpu_options=tf.GPUOptions(
    per_process_gpu_memory_fraction=0.3
))


@register('rove')
class RoVeEmbedder(TFModel):
    def __init__(self, ser_path, ser_dir='emb', ser_file='rove.model', dim=256,
                 model='lstm', rnn_size=128, num_layers=2, batch_size=32, seq_length=8,
                 letter_size=200, w2v_size=300, dropout_keep_prob=0.8, grad_clip=5.0,
                 learning_rate=0.0002, decay_rate=0.95, inference=False, train_now=True, vocab=None):
        """

        :param ser_path:
        :param ser_dir:
        :param ser_file:
        :param dim:
        :param model:
        :param rnn_size:
        :param num_layers:
        :param batch_size:
        :param seq_length:
        :param letter_size:
        :param w2v_size:
        :param dropout_keep_prob:
        :param grad_clip:
        :param learning_rate:
        :param decay_rate:
        :param train_now: bool train or inference
        :param vocab: vocabulary to inference mode from TexLoader
        """
        self.dim = dim
        self._ser_dir = ser_dir
        self._saver = None
        self.sess = None

        self.rnn_size = rnn_size
        self.w2v_size = w2v_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.letter_size = letter_size
        self.dropout = dropout_keep_prob
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.train_now = train_now
        self.num_layers = num_layers
        self.grad_clip = grad_clip
        self.ser_file = ser_file
        self.vocab = vocab

        if not train_now:
            self.batch_size = 1
            self.seq_length = 1

        if model == 'rnn':
            self.cell_fn = rnn_cell.BasicRNNCell
        elif model == 'gru':
            self.cell_fn = rnn_cell.GRUCell
        elif model == 'lstm':
            self.cell_fn = rnn_cell.BasicLSTMCell
        elif model == 'sru':
            self.cell_fn = SRUCell
        else:
            raise Exception("model type not supported: {}".format(model))
        self.load()

    def _add_placeholders(self):
        self.input_data = tf.placeholder(tf.float32, [self.batch_size, self.seq_length, self.letter_size])
        self.change = tf.placeholder(tf.bool, [self.batch_size]) # need to remove
        with tf.variable_scope("rnn", reuse=True):
            self.cell = self.cell_fn(self.rnn_size, state_is_tuple=False)
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        self.cell = rnn_cell.MultiRNNCell([self.cell] * self.num_layers, state_is_tuple=False)

    def run_sess(self, input_size, output_size):
        self._add_placeholders()

        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        initial_state = self.initial_state
        inputs = tf.split(self.input_data, self.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("input_linear"):
            linears = []
            for i, _input in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope()
                full_vector = tf.contrib.layers.fully_connected(_input, self.rnn_size,
                                                                activation_fn=None)
                linears.append(full_vector)
        fixed_input = tf.stack(linears, axis=1)
        fixed_input = tf.reshape(fixed_input, [self.batch_size, self.seq_length, -1])

        outputs, last_state = tf.nn.dynamic_rnn(self.cell, fixed_input,
                                                initial_state=initial_state,scope="rnnlm")

        self.final_state = last_state

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []

        ones = tf.diag([1.] * self.batch_size)
        outputs = tf.unstack(outputs, axis=1)
        with tf.variable_scope("output_linear"):
            for i in range(len(outputs)):
                if i > 0:
                    tf.get_variable_scope()
                output = tf.contrib.layers.fully_connected(outputs[i], self.w2v_size,
                                                           activation_fn=None)

                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, self.dropout)
                # negative sampling
                matrix = tf.matmul(output, output, transpose_b=True) - ones
                loss1 += tf.maximum(0.0, matrix)
                final_vectors.append(output)

        seq_slices = tf.reshape(tf.concat(final_vectors, 1), [self.batch_size, self.seq_length, self.w2v_size])
        seq_slices = tf.split(seq_slices, self.batch_size, 0)
        seq_slices = [tf.squeeze(input_, [0]) for input_ in seq_slices]

        with tf.variable_scope("additional_loss"):
            for i in range(len(seq_slices)):  # should be length of batch_size
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                seq_context = tf.nn.l2_normalize(seq_slices[i], 1)
                # context similarity
                matrix = tf.matmul(seq_context, seq_context, transpose_b=True)
                loss2 += 1. - matrix

        self.target = final_vectors[-1]
        self.cost = tf.reduce_sum(loss1) / self.batch_size / self.seq_length
        self.cost += tf.reduce_sum(loss2) / self.batch_size / self.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                                  self.grad_clip)
            self._optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = self._optimizer.apply_gradients(zip(grads, tvars))

        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)
        self._saver = tf.train.Saver(tf.global_variables())

    def _set_state(self, state):
        self.initial_state = state

    def _train_step(self, batch, change):
        if self.initial_state is None:
            state = self.initial_state
        else:
            state = self.initial_state.eval(session=self.sess)

        feed = {self.input_data: batch,
                self.change: change,
                self.initial_state: state}
        train_loss, state, _ = self.sess.run([self.cost, self.final_state, self.train_op], feed)
        self._set_state(state)
        return train_loss

    def _forward(self, batch, change):
        if self.initial_state is None:
            state = self.initial_state
        else:
            state = self.initial_state.eval(session=self.sess)

        feed = {self.input_data: batch,
                self.change: change,
                self.initial_state: state}
        train_loss, state, _ = self.sess.run([self.cost, self.final_state, self.train_op], feed)
        self._set_state(state)
        return train_loss

    def _encode(self, sentence: str):
        state = self.cell.zero_state(1, tf.float32).eval(session=self.sess)

        tokens = word_tokenize(sentence)
        vectors = []
        for token in tokens:
            x = self._letters2vec(token, self.vocab).reshape((1, 1, -1))
            feed = {self.input_data: x,
                    self.initial_state: state,
                    self.change: np.zeros((1,))
                    }
            [state, target] = self.sess.run([self.final_state, self.target], feed)
            vectors.append(np.squeeze(target))

        embs = np.array(vectors)
        return embs

    @check_attr_true('train_now')
    def train(self, data_loader, **kwargs):
        print(':: creating new  model\n')
        self._train_step(data_loader)
        self.save()

    def infer(self, sentence: str, *args, **kwargs):
        return self._encode(sentence)

    def load(self):
        self.run_sess(300,300) # remove input size
        if not self.train_now:
            ckpt = tf.train.get_checkpoint_state(self._ser_dir)
            self._saver.restore(self.sess, ckpt.model_checkpoint_path)

    def save(self):
        self._saver.save(self.sess, os.path.join(self._ser_dir, self.ser_file))
        print(':: model saved to path')

    def _letters2vec(self, word, vocab, dtype=np.uint8):
        base = np.zeros(len(vocab), dtype=dtype)

        def update_vector(vector, char):
            if char in vocab:
                vector[vocab.get(char, 0)] += 1

        middle = np.copy(base)
        for char in word:
            update_vector(middle, char)

        first = np.copy(base)
        update_vector(first, word[0])
        second = np.copy(base)
        if len(word) > 1:
            update_vector(second, word[1])
        third = np.copy(base)
        if len(word) > 2:
            update_vector(third, word[2])

        end_first = np.copy(base)
        update_vector(end_first, word[-1])
        end_second = np.copy(base)
        if len(word) > 1:
            update_vector(end_second, word[-2])
        end_third = np.copy(base)
        if len(word) > 2:
            update_vector(end_third, word[-3])
        return np.concatenate([first, second, third, middle, end_third, end_second, end_first])

    def shutdown(self):
        self.sess.close()
