import numpy as np
import tensorflow as tf

from deeppavlov.core.layers.tf_layers import embedding_layer
from deeppavlov.core.models.nn_model import NNModel


class NerNetwork(NNModel):
    def __init__(self,
                 capitalization_dim=None,  # Features dimensions
                 pos_features_dim=None,
                 net_type='rnn',  # Net architecture
                 cell_type='lstm',
                 use_cudnn_rnn=False,
                 two_top_dense=False,
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
        self._add_training_placeholders()
        self._xs_and_y_placeholders = []
        self._input_features = []

        # ---------------  Building input features ---------------

        # Token embeddings
        self._build_word_embeddings(token_emb_dim, token_emb_mat, embeddings_dropout)

        # Char embeddings using highway CNN with max pooling
        if self.char_emb_dim is not None:
            self._build_char_embeddings(char_emb_dim, char_emb_mat, embeddings_dropout)

        # Capitalization features
        if capitalization_dim is not None:
            self._build_capitalization(token_emb_dim, token_emb_mat)

        # Part of speech features
        if pos_features_dim is not None:
            self._build_pos(pos_features_dim)

        features = tf.concat(self._input_features)

        # ----------------  Building the network ----------------

        if net_type == 'rnn':
            if use_cudnn_rnn:
                units = self._build_cudnn_rnn(features, n_hidden_list, cell_type, intra_layer_dropout, l2_reg)
            else:
                units = self._build_rnn(features, n_hidden_list, cell_type, intra_layer_dropout, l2_reg)
        elif net_type == 'cnn':
            units = self._build_cnn(features, n_hidden_list, cnn_filter_width, use_batch_norm,  l2_reg)

        logits = self._build_top(units, n_hidden_list[-1], top_dropout, two_top_dense)

        self.train_op, self.predict = self._build_train_crf(logits, use_crf)

        # ----------- Initilize th session --------------

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        if gpu is not None:
            sess_config.gpu_options.visible_device_list = str(gpu)

        self.sess = tf.Session(sess_config)

    def _add_training_placeholders(self):
        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        self.dropout_ph = tf.placeholder_with_default(1.0, shape=[], name='dropout')
        self.training_ph = tf.placeholder_with_default(False, shape=[], name='is_training')

    @staticmethod
    def _build_embeddings(indices, n_tokens, embedding_mat=None, dropout_ph=None):
        pass
