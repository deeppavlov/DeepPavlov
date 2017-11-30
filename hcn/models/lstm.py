from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.python.training.saver import Saver
from tensorflow.contrib.training import HParams
from tensorflow.contrib.layers import xavier_initializer as xav

from deeppavlov.common import paths
from deeppavlov.common.registry import register_model
from deeppavlov.models.tf_model import TFModel

config = tf.ConfigProto(
    device_count={'GPU': 0}
)


@register_model('lstm')
class LSTM(TFModel):
    def __init__(self, input_size, output_size, num_hidden_units=128,
                 optimizer=tf.train.AdadeltaOptimizer(0.1), saver=Saver,
                 model_dir_path='ckpt/', model_fpath='hcn.ckpt'):
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = optimizer
        self._saver = saver
        self._hps = HParams(num_hidden_units=num_hidden_units)
        self._model_path = Path(paths.USR_PATH).joinpath(model_dir_path, model_fpath)

        self._run_sess()

    def _add_placeholders(self):
        self._features = tf.placeholder(tf.float32, [1, self.input_size], name='input_features')
        self.init_state_c_ = tf.placeholder(tf.float32, [1, self._hps.num_hidden_units])
        self.init_state_h_ = tf.placeholder(tf.float32, [1, self._hps.num_hidden_units])
        self._action = tf.placeholder(tf.int32, name='ground_truth_action')
        self._action_mask = tf.placeholder(tf.float32, [self.output_size], name='action_mask')

    def _set_state(self, state_c, state_h):
        self._init_state_c = state_c
        self._init_state_h = state_h

    def _run_sess(self):
        self._build_graph()
        # input projection
        Wi = tf.get_variable('Wi', [self.input_size, self._hps.num_hidden_units],
                             initializer=xav())
        bi = tf.get_variable('bi', [self._hps.num_hidden_units],
                             initializer=tf.constant_initializer(0.))

        # add relu/tanh here if necessary
        projected_features = tf.matmul(self._features, Wi) + bi

        lstm_f = tf.contrib.rnn.LSTMCell(self._hps.num_hidden_units, state_is_tuple=True)

        lstm_op, state = lstm_f(inputs=projected_features,
                                state=(self.init_state_c_, self.init_state_h_))

        # reshape LSTM's state tuple (2,128) -> (1,256)
        state_reshaped = tf.concat(axis=1, values=(state.c, state.h))

        # output projection
        Wo = tf.get_variable('Wo', [2 * self._hps.num_hidden_units, self.output_size],
                             initializer=xav())
        bo = tf.get_variable('bo', [self.output_size],
                             initializer=tf.constant_initializer(0.))
        # get logits
        logits = tf.matmul(state_reshaped, Wo) + bo
        # probabilities
        #  normalization : elemwise multiply with action mask
        probs = tf.multiply(tf.squeeze(tf.nn.softmax(logits)), self._action_mask)

        # prediction
        prediction = tf.argmax(probs, axis=0)

        # loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._action)
        self._train_op = self._optimizer.minimize(loss)

        self.loss = loss
        self.prediction = prediction
        self.probs = probs
        self.logits = logits
        self.state = state

        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.reset_state()

    def _build_graph(self):
        tf.reset_default_graph()
        self._add_placeholders()

    def _train_step(self, features, action, action_mask):
        _, loss_value, state_c, state_h = self.sess.run(
            [self._train_op, self.loss, self.state.c, self.state.h],
            feed_dict={
                self._features: features.reshape([1, self.input_size]),
                self._action: [action],
                self.init_state_c_: self._init_state_c,
                self.init_state_h_: self._init_state_h,
                self._action_mask: action_mask
            })
        self._set_state(state_c, state_h)
        return loss_value

    def _forward(self, features, action_mask):
        probas, prediction, state_c, state_h = self.sess.run(
            [self.probs, self.prediction, self.state.c, self.state.h],
            feed_dict={
                self._features: features.reshape([1, self.input_size]),
                self.init_state_c_: self._init_state_c,
                self.init_state_h_: self._init_state_h,
                self._action_mask: action_mask
            })
        self._set_state(state_c, state_h)

        # return argmax
        return prediction

    def reset_state(self):
        init_state = np.zeros([1, self._hps.num_hidden_units], dtype=np.float32)
        self._set_state(init_state, init_state)
