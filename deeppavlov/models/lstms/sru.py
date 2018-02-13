import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


class SRUCell(RNNCell):
    """Simple recurrent unit cell.
    The implementation of: https://arxiv.org/abs/1709.02755.
    """
    def __init__(self, num_units, state_is_tuple=True, activation=tf.nn.tanh, reuse=None):
        super(SRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._linear = None
        self._W = tf.Variable(self.init_matrix([self._num_units, 3 * self._num_units]))
        self._bias = tf.Variable(self.init_matrix([2 * self._num_units]))

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """
        f - forget gate
        r - reset gate
        c - final cell
        :param inputs:
        :param state:
        :param scope:
        :return: state, cell
        """
        with variable_scope.variable_scope(scope or type(self).__name__):
            U = math_ops.matmul(inputs, self._W)

            x_in, f_resource, r_resource = array_ops.split(value=U,
                                                            num_or_size_splits=3,
                                                            axis=1)
            f_r = math_ops.sigmoid(nn_ops.bias_add(array_ops.concat(
                 [f_resource, r_resource], 1), self._bias))
            f, r = array_ops.split(value=f_r, num_or_size_splits=2, axis=1)
            c = f * state + (1.0 - f) * x_in
            hidden_state = r * self._activation(c) + (1.0 - r) * inputs

            if self._state_is_tuple:
                return hidden_state, LSTMStateTuple(c, hidden_state)
            else:
                return hidden_state, c

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)
