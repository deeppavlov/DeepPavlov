import numpy as np
from gensim.models import word2vec

import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
rnn_cell = rnn_cell_impl
rnn = tf.contrib.rnn

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.attributes import check_attr_true


@register('rove')
class RoVeEmbedder(TFModel):
    def _train_step(self, features, *args):
        pass

    def __init__(self, ser_path, ser_dir='emb', ser_file='text8.model', dim=256,
                 model='lstm', rnn_size=128, num_layers=2, batch_size=32, seq_length=8,
                 letter_size=200, w2v_size=300, dropout_keep_prob=0.8, grad_clip=5.0,
                 train_now=False):
        super().__init__(ser_path=ser_path, ser_dir=ser_dir,
                         ser_file=ser_file, train_now=train_now)
        self.dim = dim
        self.model = self.load()

        if not train_now:
            batch_size = 1
            seq_length = 1

        if model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(model))

        #with tf.variable_scope("rnn", reuse=True):
        cell = cell_fn(rnn_size, state_is_tuple=False)# is not necesery arg

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=False)
        self.input_data = tf.placeholder(tf.float32, [batch_size, seq_length, letter_size])
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        self.change = tf.placeholder(tf.bool, [batch_size])

        initial_state = self.initial_state
        inputs = tf.split(self.input_data, seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        with tf.variable_scope("input_linear"):
            linears = []

            for i in range(len(inputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                linears.append((rnn_cell._linear(inputs[i], rnn_size, bias=True)))

        outputs, last_state = rnn_decoder(linears, initial_state, cell,
                                          scope="rnnlm")
        print("Shape of the last state",last_state.shape)
        self.final_state = last_state

        loss1 = tf.constant(0.0)
        loss2 = tf.constant(0.0)
        final_vectors = []

        ones = tf.diag([1.] * batch_size)
        with tf.variable_scope("output_linear"):
            for i in range(len(outputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output = rnn_cell._linear(outputs[i], w2v_size, bias=True)

                output = tf.nn.l2_normalize(output, 1)
                output = tf.nn.dropout(output, dropout_keep_prob)
                # negative sampling
                matrix = tf.matmul(output, output, transpose_b=True) - ones
                loss1 += tf.maximum(0.0, matrix)
                final_vectors.append(output)

        seq_slices = tf.reshape(tf.concat(final_vectors, 1), [batch_size, seq_length, w2v_size])
        seq_slices = tf.split(seq_slices, batch_size, 0)
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
        self.cost = tf.reduce_sum(loss1) / batch_size / seq_length
        self.cost += tf.reduce_sum(loss2) / batch_size / seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars,
                                                           aggregation_method=
                                                           tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
                                              grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        # Validation
        self.valid_data = tf.placeholder(tf.float32, [1, 1, letter_size])
        self.valid_initial_state = cell.zero_state(1, tf.float32)

        valid_initial_state = self.valid_initial_state

        valid_inputs = tf.split(self.valid_data, 1, 1)
        valid_inputs = [tf.squeeze(input_, [1]) for input_ in valid_inputs]

        with tf.variable_scope("input_valid"):
            valid_fixed_input = []
            for i, _input in enumerate(valid_inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                valid_fixed_input.append((rnn_cell._linear(valid_inputs[i], rnn_size, bias=True)))

        valid_outputs, valid_last_state = rnn_decoder(valid_fixed_input, valid_initial_state, cell,
                                          scope="rnnlm")

        self.valid_state = valid_last_state

        valid_vectors = []

        valid_outputs = tf.unstack(valid_outputs, axis = 1)
        with tf.variable_scope("output_valid"):
            for i in range(len(valid_outputs)):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output = rnn_cell._linear(valid_outputs[i], w2v_size, bias=True)
                valid_vectors.append(output)
        self.valid_vector = valid_vectors[-1]

    def _encode(self, sentence: str):
        embs = [self.model[w] for w in sentence.split() if w and w in self.model]
        # average of embeddings
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim], np.float32)

    @check_attr_true('train_now')
    def train(self, *args, **kwargs):
        sentences = word2vec.Text8Corpus(self.ser_path)

        print(':: creating new word2vec model')
        model = word2vec.Word2Vec(sentences, size=self.dim)
        self.model = model

        self.ser_path.parent.mkdir(parents=True, exist_ok=True)
        self.save()
        return model

    def infer(self, sentence: str, *args, **kwargs):
        return self._encode(sentence)

    # @run_alt_meth_if_no_path(train, 'train_now')
    def load(self):
        return word2vec.Word2Vec.load(str(self.ser_path))

    def save(self):
        self.model.save(str(self.ser_path))
        print(':: model saved to path')
