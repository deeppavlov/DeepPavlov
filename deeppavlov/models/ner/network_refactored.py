import numpy as np
import tensorflow as tf

from deeppavlov.core.layers.tf_layers import embedding_layer, character_embedding_network, variational_dropout
from deeppavlov.core.layers.tf_layers import cudnn_bi_lstm, cudnn_bi_gru, stacked_bi_rnn
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.utils import check_gpu_existance


class NerNetwork(NNModel):
    def __init__(self,
                 capitalization_dim=None,  # Features dimensions
                 pos_features_dim=None,
                 net_type='rnn',  # Net architecture
                 cell_type='lstm',
                 use_cudnn_rnn=False,
                 two_dense_on_top=False,
                 n_hidden_list=(128,),
                 cnn_filter_width=7,
                 use_crf=False,
                 token_emb_mat=None,
                 char_emb_mat=None,
                 concat_bi=False,
                 use_batch_norm=False,  # Regularization
                 embeddings_dropout=False,
                 top_dropout=False,
                 intra_layer_dropout=False,
                 l2_reg=0.0,
                 gpu=None):
        self._build_training_placeholders()
        self._xs_and_y_placeholders = []
        self._input_features = []

        # ================ Building input features =================

        # Token embeddings
        self._build_word_embeddings(token_emb_mat, embeddings_dropout)

        # Char embeddings using highway CNN with max pooling
        if self.char_emb_dim is not None:
            self._build_char_embeddings(char_emb_mat, embeddings_dropout)

        # Capitalization features
        if capitalization_dim is not None:
            self._build_capitalization(capitalization_dim)

        # Part of speech features
        if pos_features_dim is not None:
            self._build_pos(pos_features_dim)

        features = tf.concat(self._input_features)

        # ================== Building the network ==================

        if net_type == 'rnn':
            if use_cudnn_rnn:
                units = self._build_cudnn_rnn(features, n_hidden_list, cell_type, intra_layer_dropout)
            else:
                units = self._build_rnn(features, n_hidden_list, cell_type, intra_layer_dropout)
        elif net_type == 'cnn':
            units = self._build_cnn(features, n_hidden_list, cnn_filter_width, use_batch_norm)

        logits = self._build_top(units, n_hidden_list[-1], top_dropout, two_dense_on_top)

        self.train_op, self.loss, predict_method = self._build_train_predict(logits, use_crf, l2_reg)
        self.predict = predict_method

        # ================= Initialize the session =================

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        if gpu is not None:
            sess_config.gpu_options.visible_device_list = str(gpu)

        self.sess = tf.Session(sess_config)

    def _build_training_placeholders(self):
        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        self._dropout_ph = tf.placeholder_with_default(1.0, shape=[], name='dropout')
        self.training_ph = tf.placeholder_with_default(False, shape=[], name='is_training')

    def _build_word_embeddings(self, token_emb_mat, embeddings_dropout):
        token_indices_ph = tf.placeholder(tf.int32, [None, None])
        emb = embedding_layer(token_indices_ph, token_emb_mat)
        if embeddings_dropout:
            emb = tf.layers.dropout(emb, self._dropout_ph, noise_shape=[tf.shape(emb)[0], 1, tf.shape(emb)[2]])
        self._xs_and_y_placeholders.append(token_indices_ph)
        self._input_features.append(emb)

    def _build_char_embeddings(self, char_emb_mat, embeddings_dropout):
        character_indices_ph = tf.placeholder(tf.int32, [None, None, None], name='Char_PH')
        character_embedding_network()

    def _build_capitalization(self, capitalization_dim):
        capitalization_ph = tf.placeholder(tf.int32, [None, None, capitalization_dim], name='Capitalization_PH')
        self._xs_and_y_placeholders.append(capitalization_ph)
        self._input_features.append(capitalization_ph)

    def _build_pos(self, pos_features_dim):
        pos_ph = tf.placeholder(tf.int32, [None, None, pos_features_dim], name='POS_PH')
        self._xs_and_y_placeholders.append(pos_ph)
        self._input_features.append(pos_ph)

    def _build_cudnn_rnn(self, units, n_hidden_list, cell_type, intra_layer_dropout):
        if not check_gpu_existance():
            raise RuntimeError('Usage of cuDNN RNN layers require GPU along with cuDNN library')
        for n, n_hidden in enumerate(n_hidden_list):
            with tf.variable_scope(cell_type.upper() + '_' + str(n)):
                if cell_type.lower() == 'lstm':
                    units, _ = cudnn_bi_lstm(units, n_hidden)
                elif cell_type.lower() == 'gru':
                    units, _ = cudnn_bi_gru(units, n_hidden)
                else:
                    raise RuntimeError('Wrong cell type "{}"! Only "gru" and "lstm"!'.format(cell_type))
                units = tf.concat(units, -1)
                if intra_layer_dropout:
                    units = variational_dropout(units, self._dropout_ph)

    def _build_rnn(self, units, n_hidden_list, cell_type, intra_layer_dropout, l2_reg):
        for n, n_hidden in enumerate(n_hidden_list):
            with tf.variable_scope(cell_type.upper() + '_' + str(n)):
                if cell_type.lower() == 'lstm':
                    units, _ = cudnn_bi_lstm(units, n_hidden)
                elif cell_type.lower() == 'gru':
                    units, _ = cudnn_bi_gru(units, n_hidden)
                else:
                    raise RuntimeError('Wrong cell type "{}"! Only "gru" and "lstm"!'.format(cell_type))
                if intra_layer_dropout:
                    units = variational_dropout(units, self._dropout_ph)

    @staticmethod
    def _build_cnn(features, n_hidden_list, cnn_filter_width, use_batch_norm,  l2_reg):
        pass

    @staticmethod
    def _build_top(units, n_hididden, top_dropout, two_dense_on_top):
        pass

    @staticmethod
    def _build_train_predict(logits, use_crf):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def train_on_batch(self, x: list, y: list):
        pass

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass