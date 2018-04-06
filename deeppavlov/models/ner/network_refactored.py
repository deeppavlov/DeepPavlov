import numpy as np
import tensorflow as tf

from deeppavlov.core.layers.tf_layers import embedding_layer, character_embedding_network
from deeppavlov.core.models.nn_model import NNModel


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
                 char_emb_dim=None,  # Embeddings parameters
                 token_emb_dim=100,
                 token_emb_mat=None,
                 char_emb_mat=None,
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
        self._build_word_embeddings(token_emb_dim, token_emb_mat, embeddings_dropout)

        # Char embeddings using highway CNN with max pooling
        if self.char_emb_dim is not None:
            self._build_char_embeddings(char_emb_dim, char_emb_mat, embeddings_dropout)

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
                units = self._build_cudnn_rnn(features, n_hidden_list, cell_type, intra_layer_dropout, l2_reg)
            else:
                units = self._build_rnn(features, n_hidden_list, cell_type, intra_layer_dropout, l2_reg)
        elif net_type == 'cnn':
            units = self._build_cnn(features, n_hidden_list, cnn_filter_width, use_batch_norm, l2_reg)

        logits = self._build_top(units, n_hidden_list[-1], top_dropout, two_dense_on_top)

        self.train_op, self.loss, predict_method = self._build_train_predict(logits, use_crf)
        self.predict = predict_method

        # ================= Initialize the session =================

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        if gpu is not None:
            sess_config.gpu_options.visible_device_list = str(gpu)

        self.sess = tf.Session(sess_config)

    def _build_training_placeholders(self):
        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        self.dropout_ph = tf.placeholder_with_default(1.0, shape=[], name='dropout')
        self.training_ph = tf.placeholder_with_default(False, shape=[], name='is_training')

    def _build_word_embeddings(self, token_emb_dim, token_emb_mat, embeddings_dropout):
        token_indices_ph = tf.placeholder(tf.int32, [None, None])
        emb = embedding_layer(token_indices_ph, token_emb_mat, token_embedding_dim=token_emb_dim)
        if embeddings_dropout:
            emb = tf.layers.dropout(emb, self.dropout_ph, noise_shape=[tf.shape(emb)[0], 1, tf.shape(emb)[2]])
        self._xs_and_y_placeholders.append(token_indices_ph)
        self._input_features.append(emb)

    def _build_char_embeddings(self, char_emb_dim, char_emb_mat, embeddings_dropout):
        character_indices_ph = tf.placeholder(tf.int32, [None, None, None])
        character_embedding_network()

    def _build_capitalization(self, capitalization_dim):
        pass

    def _build_pos(self, pos_features_dim):
        pass

    @staticmethod
    def _build_cudnn_rnn(features, n_hidden_list, cell_type, intra_layer_dropout, l2_reg):
        pass

    @staticmethod
    def _build_rnn(features, n_hidden_list, cell_type, intra_layer_dropout, l2_reg):
        pass

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