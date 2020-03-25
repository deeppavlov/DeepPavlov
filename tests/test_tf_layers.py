import shutil
from functools import reduce
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from deeppavlov.core.layers.tf_layers import cudnn_lstm, cudnn_compatible_lstm, cudnn_gru, cudnn_compatible_gru

tests_dir = Path(__file__).parent
tf_layers_data_path = tests_dir / "tf_layers_data"


def setup_module():
    shutil.rmtree(str(tf_layers_data_path), ignore_errors=True)
    tf_layers_data_path.mkdir(parents=True)


def teardown_module():
    shutil.rmtree(str(tf_layers_data_path), ignore_errors=True)


class DPCudnnLSTMModel:
    def __init__(self, num_layers, num_units):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self.x = tf.placeholder(shape=(None, None, 50), dtype=tf.float32)
        with tf.variable_scope('cudnn_model'):
            h, (h_last, c_last) = cudnn_lstm(self.x, num_units, num_layers, trainable_initial_states=True)

            self.h = h
            self.h_last = h_last

        self.sess.run(tf.global_variables_initializer())

    def __call__(self, x):
        feed_dict = {
            self.x: x,
        }
        return self.sess.run([self.h, self.h_last], feed_dict=feed_dict)

    def save(self, path='model'):
        print('[saving model to {}]'.format(path))
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


class DPLSTMModel:
    def __init__(self, num_layers, num_units):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self.x = tf.placeholder(shape=(None, None, 50), dtype=tf.float32)
        with tf.variable_scope('cudnn_model'):
            h, (h_last, c_last) = cudnn_compatible_lstm(self.x, num_units, num_layers, trainable_initial_states=True)

            self.h = h
            self.h_last = h_last

        self.sess.run(tf.global_variables_initializer())

    def __call__(self, x):
        feed_dict = {
            self.x: x,
        }
        return self.sess.run([self.h, self.h_last], feed_dict=feed_dict)

    def save(self, path='model'):
        print('[saving model to {}]'.format(path))
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


class DPCudnnGRUModel:
    def __init__(self, num_layers, num_units):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self.x = tf.placeholder(shape=(None, None, 50), dtype=tf.float32)
        with tf.variable_scope('cudnn_model'):
            h, h_last = cudnn_gru(self.x, num_units, num_layers, trainable_initial_states=True)

            self.h = h
            self.h_last = h_last

        self.sess.run(tf.global_variables_initializer())

    def __call__(self, x):
        feed_dict = {
            self.x: x,
        }
        return self.sess.run([self.h, self.h_last], feed_dict=feed_dict)

    def save(self, path='model'):
        print('[saving model to {}]'.format(path))
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


class DPGRUModel:
    def __init__(self, num_layers, num_units):
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self.x = tf.placeholder(shape=(None, None, 50), dtype=tf.float32)
        with tf.variable_scope('cudnn_model'):
            h, h_last = cudnn_compatible_gru(self.x, num_units, num_layers, trainable_initial_states=True)

            self.h = h
            self.h_last = h_last

        self.sess.run(tf.global_variables_initializer())

    def __call__(self, x):
        feed_dict = {
            self.x: x,
        }
        return self.sess.run([self.h, self.h_last], feed_dict=feed_dict)

    def save(self, path='model'):
        print('[saving model to {}]'.format(path))
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


class TestTFLayers:
    allowed_error_lvl = 0.01 * 2 ** 0.5

    @staticmethod
    def equal_values(a, b, round=5):
        a, b = np.round(a, round), np.round(b, round)
        return np.sum(a == b) / reduce(lambda x, y: x * y, a.shape)

    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_cudnn_lstm_save_load(self, num_layers):
        x = np.random.normal(size=(10, 10, 50))
        tf.reset_default_graph()
        cdnnlstmmodel = DPCudnnLSTMModel(num_layers=num_layers, num_units=100)
        before_load_hidden, before_load_state = cdnnlstmmodel(x)[0], cdnnlstmmodel(x)[1]
        cdnnlstmmodel.save(str(tf_layers_data_path / 'dpcudnnlstmmodel' / 'model'))

        tf.reset_default_graph()
        cdnnlstmmodel = DPCudnnLSTMModel(num_layers=num_layers, num_units=100)
        cdnnlstmmodel.load(str(tf_layers_data_path / 'dpcudnnlstmmodel' / 'model'))
        after_load_hidden, after_load_state = cdnnlstmmodel(x)[0], cdnnlstmmodel(x)[1]

        equal_hidden = self.equal_values(after_load_hidden, before_load_hidden)
        equal_state = self.equal_values(after_load_state, before_load_state)

        assert equal_hidden > 1 - self.allowed_error_lvl
        assert equal_state > 1 - self.allowed_error_lvl

    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_cudnn_lstm_save_and_cudnn_compatible_load(self, num_layers):
        x = np.random.normal(size=(10, 10, 50))
        tf.reset_default_graph()
        cdnnlstmmodel = DPCudnnLSTMModel(num_layers=num_layers, num_units=100)
        before_load_hidden, before_load_state = cdnnlstmmodel(x)[0], cdnnlstmmodel(x)[1]
        cdnnlstmmodel.save(str(tf_layers_data_path / 'dpcudnnlstmmodel' / 'model'))

        tf.reset_default_graph()
        cdnnlstmmodel = DPLSTMModel(num_layers=num_layers, num_units=100)
        cdnnlstmmodel.load(str(tf_layers_data_path / 'dpcudnnlstmmodel' / 'model'))
        after_load_hidden, after_load_state = cdnnlstmmodel(x)[0], cdnnlstmmodel(x)[1]

        equal_hidden = self.equal_values(after_load_hidden, before_load_hidden)
        equal_state = self.equal_values(after_load_state, before_load_state)

        assert equal_hidden > 1 - self.allowed_error_lvl
        assert equal_state > 1 - self.allowed_error_lvl

    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_cudnn_gru_save_load(self, num_layers):
        x = np.random.normal(size=(10, 10, 50))
        tf.reset_default_graph()
        cdnngrumodel = DPCudnnGRUModel(num_layers=num_layers, num_units=100)
        before_load_hidden, before_load_state = cdnngrumodel(x)[0], cdnngrumodel(x)[1]
        cdnngrumodel.save(str(tf_layers_data_path / 'cdnngrumodel' / 'model'))

        tf.reset_default_graph()
        cdnngrumodel = DPCudnnGRUModel(num_layers=num_layers, num_units=100)
        cdnngrumodel.load(str(tf_layers_data_path / 'cdnngrumodel' / 'model'))
        after_load_hidden, after_load_state = cdnngrumodel(x)[0], cdnngrumodel(x)[1]

        equal_hidden = self.equal_values(after_load_hidden, before_load_hidden)
        equal_state = self.equal_values(after_load_state, before_load_state)

        assert equal_hidden > 1 - self.allowed_error_lvl
        assert equal_state > 1 - self.allowed_error_lvl

    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_cudnn_gru_save_and_cudnn_compatible_load(self, num_layers):
        x = np.random.normal(size=(10, 10, 50))
        tf.reset_default_graph()
        cdnngrumodel = DPCudnnGRUModel(num_layers=num_layers, num_units=100)
        before_load_hidden, before_load_state = cdnngrumodel(x)[0], cdnngrumodel(x)[1]
        cdnngrumodel.save(str(tf_layers_data_path / 'cdnngrumodel' / 'model'))

        tf.reset_default_graph()
        cdnngrumodel = DPGRUModel(num_layers=num_layers, num_units=100)
        cdnngrumodel.load(str(tf_layers_data_path / 'cdnngrumodel' / 'model'))
        after_load_hidden, after_load_state = cdnngrumodel(x)[0], cdnngrumodel(x)[1]

        equal_hidden = self.equal_values(after_load_hidden, before_load_hidden)
        equal_state = self.equal_values(after_load_state, before_load_state)

        assert equal_hidden > 1 - self.allowed_error_lvl
        assert equal_state > 1 - self.allowed_error_lvl
