import numpy as np
import tensorflow as tf
from functools import partial

from deeppavlov.core.layers.tf_layers import embedding_layer, character_embedding_network, variational_dropout
from deeppavlov.core.layers.tf_layers import cudnn_bi_lstm, cudnn_bi_gru, bi_rnn, stacked_cnn, INITIALIZER
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.check_gpu import check_gpu_existence
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('ner')
class NerNetwork(TFModel):
    GRAPH_PARAMS = ["n_tags",  # TODO: add check
                    "char_emb_dim",
                    "capitalization_dim",
                    "additional_features",
                    "use_char_embeddings",
                    "additional_features",
                    "net_type",
                    "cell_type",
                    "char_filter_width",
                    "cell_type"]

    def __init__(self,
                 n_tags,  # Features dimensions
                 token_emb_dim=None,
                 char_emb_dim=None,
                 capitalization_dim=None,
                 pos_features_dim=None,
                 additional_features=None,
                 net_type='rnn',  # Net architecture
                 cell_type='lstm',
                 use_cudnn_rnn=False,
                 two_dense_on_top=False,
                 n_hidden_list=(128,),
                 cnn_filter_width=7,
                 use_crf=False,
                 token_emb_mat=None,
                 char_emb_mat=None,
                 use_batch_norm=False,
                 dropout_keep_prob=0.5,  # Regularization
                 embeddings_dropout=False,
                 top_dropout=False,
                 intra_layer_dropout=False,
                 l2_reg=0.0,
                 clip_grad_norm=5.0,
                 learning_rate=3e-3,
                 gpu=None,
                 seed=None,
                 lr_drop_patience=5,
                 lr_drop_value=0.1,
                 **kwargs):
        tf.set_random_seed(seed)
        np.random.seed(seed)
        self._learning_rate = learning_rate
        self._lr_drop_patience = lr_drop_patience
        self._lr_drop_value = lr_drop_value
        self._add_training_placeholders(dropout_keep_prob, learning_rate)
        self._xs_ph_list = []
        self._y_ph = tf.placeholder(tf.int32, [None, None], name='y_ph')
        self._input_features = []

        # ================ Building input features =================

        # Token embeddings
        self._add_word_embeddings(token_emb_mat, token_emb_dim)

        # Masks for different lengths utterances
        self.mask_ph = self._add_mask()

        # Char embeddings using highway CNN with max pooling
        if char_emb_mat is not None and char_emb_dim is not None:
            self._add_char_embeddings(char_emb_mat)

        # Capitalization features
        if capitalization_dim is not None:
            self._add_capitalization(capitalization_dim)

        # Part of speech features
        if pos_features_dim is not None:
            self._add_pos(pos_features_dim)

        # Anything you want
        if additional_features is not None:
            self._add_additional_features(additional_features)

        features = tf.concat(self._input_features, axis=2)
        if embeddings_dropout:
            features = variational_dropout(features, self._dropout_ph)

        # ================== Building the network ==================

        if net_type == 'rnn':
            if use_cudnn_rnn:
                if l2_reg > 0:
                    raise Warning('cuDNN RNN are not l2 regularizable')
                units = self._build_cudnn_rnn(features, n_hidden_list, cell_type, intra_layer_dropout, self.mask_ph)
            else:
                units = self._build_rnn(features, n_hidden_list, cell_type, intra_layer_dropout, self.mask_ph,)
        elif net_type == 'cnn':
            units = self._build_cnn(features, n_hidden_list, cnn_filter_width, use_batch_norm)
        self._logits = self._build_top(units, n_tags, n_hidden_list[-1], top_dropout, two_dense_on_top)

        self.train_op, self.loss = self._build_train_predict(self._logits, self.mask_ph, n_tags,
                                                             use_crf, clip_grad_norm, l2_reg)
        self.predict = self.predict_crf if use_crf else self.predict_no_crf

        # ================= Initialize the session =================

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        if gpu is not None:
            sess_config.gpu_options.visible_device_list = str(gpu)
        self.sess = tf.Session()   # TODO: add sess_config
        self.sess.run(tf.global_variables_initializer())
        super().__init__(**kwargs)
        self.load()

    def _add_training_placeholders(self, dropout_keep_prob, learning_rate):
        self.learning_rate_ph = tf.placeholder_with_default(learning_rate, shape=[], name='learning_rate')
        self._dropout_ph = tf.placeholder_with_default(dropout_keep_prob, shape=[], name='dropout')
        self.training_ph = tf.placeholder_with_default(False, shape=[], name='is_training')

    def _add_word_embeddings(self, token_emb_mat, token_emb_dim=None):
        if token_emb_mat is None:
            token_ph = tf.placeholder(tf.float32, [None, None, token_emb_dim], name='Token_Ind_ph')
            emb = token_ph
        else:
            token_ph = tf.placeholder(tf.int32, [None, None], name='Token_Ind_ph')
            emb = embedding_layer(token_ph, token_emb_mat)
        self._xs_ph_list.append(token_ph)
        self._input_features.append(emb)

    def _add_mask(self):
        mask_ph = tf.placeholder(tf.float32, [None, None], name='Mask_ph')
        self._xs_ph_list.append(mask_ph)
        return mask_ph

    def _add_char_embeddings(self, char_emb_mat):
        character_indices_ph = tf.placeholder(tf.int32, [None, None, None], name='Char_ph')
        char_embs = character_embedding_network(character_indices_ph, emb_mat=char_emb_mat)
        self._xs_ph_list.append(character_indices_ph)
        self._input_features.append(char_embs)

    def _add_capitalization(self, capitalization_dim):
        capitalization_ph = tf.placeholder(tf.float32, [None, None, capitalization_dim], name='Capitalization_ph')
        self._xs_ph_list.append(capitalization_ph)
        self._input_features.append(capitalization_ph)

    def _add_pos(self, pos_features_dim):
        pos_ph = tf.placeholder(tf.float32, [None, None, pos_features_dim], name='POS_ph')
        self._xs_ph_list.append(pos_ph)
        self._input_features.append(pos_ph)

    def _add_additional_features(self, features_list):
        for feature, dim in features_list:
            feat_ph = tf.placeholder(tf.float32, [None, None, dim], name=feature + '_ph')
            self._xs_ph_list.append(feat_ph)
            self._input_features.append(feat_ph)

    def _build_cudnn_rnn(self, units, n_hidden_list, cell_type, intra_layer_dropout, mask):
        if not check_gpu_existence():
            raise RuntimeError('Usage of cuDNN RNN layers require GPU along with cuDNN library')
        sequence_lengths = tf.to_int32(tf.reduce_sum(mask, axis=1))
        for n, n_hidden in enumerate(n_hidden_list):
            with tf.variable_scope(cell_type.upper() + '_' + str(n)):
                if cell_type.lower() == 'lstm':
                    units, _ = cudnn_bi_lstm(units, n_hidden, sequence_lengths)
                elif cell_type.lower() == 'gru':
                    units, _ = cudnn_bi_gru(units, n_hidden, sequence_lengths)
                else:
                    raise RuntimeError('Wrong cell type "{}"! Only "gru" and "lstm"!'.format(cell_type))
                units = tf.concat(units, -1)
                if intra_layer_dropout and n != len(n_hidden_list) - 1:
                    units = variational_dropout(units, self._dropout_ph)
            return units

    def _build_rnn(self, units, n_hidden_list, cell_type, intra_layer_dropout):
        for n, n_hidden in enumerate(n_hidden_list):
            units, _ = bi_rnn(units, n_hidden, cell_type=cell_type, name='Layer_' + str(n))
            units = tf.concat(units, -1)
            if intra_layer_dropout and n != len(n_hidden_list) - 1:
                units = variational_dropout(units, self._dropout_ph)
        return units

    def _build_cnn(self, units, n_hidden_list, cnn_filter_width, use_batch_norm):
        units = stacked_cnn(units, n_hidden_list, cnn_filter_width, use_batch_norm, training_ph=self.training_ph)
        return units

    def _build_top(self, units, n_tags, n_hididden, top_dropout, two_dense_on_top):
        if top_dropout:
            units = variational_dropout(units, self._dropout_ph)
        if two_dense_on_top:
            units = tf.layers.dense(units, n_hididden, activation=tf.nn.relu,
                                    kernel_initializer=INITIALIZER(),
                                    kernel_regularizer=tf.nn.l2_loss)
        logits = tf.layers.dense(units, n_tags, activation=None,
                                 kernel_initializer=INITIALIZER(),
                                 kernel_regularizer=tf.nn.l2_loss)
        return logits

    def _build_train_predict(self, logits, mask, n_tags, use_crf, clip_grad_norm, l2_reg):
        if use_crf:
            sequence_lengths = tf.reduce_sum(mask, axis=1)
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, self._y_ph, sequence_lengths)
            loss_tensor = -log_likelihood
            self._transition_params = transition_params
        else:
            ground_truth_labels = tf.one_hot(self._y_ph, n_tags)
            loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_labels, logits=logits)
            loss_tensor = loss_tensor * mask
            self._y_pred = tf.argmax(logits, axis=-1)

        loss = tf.reduce_mean(loss_tensor)

        # L2 regularization
        if l2_reg > 0:
            loss += l2_reg * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # optimizer = partial(tf.train.MomentumOptimizer, momentum=0.9, use_nesterov=True)
        optimizer = tf.train.AdamOptimizer
        train_op = self.get_train_op(loss, self.learning_rate_ph, optimizer, clip_norm=clip_grad_norm)
        return train_op, loss

    def predict_no_crf(self, xs):
        feed_dict = self._fill_feed_dict(xs)
        pred_idxs, mask = self.sess.run([self._y_pred, self.mask_ph], feed_dict)

        # Filter by sequece length
        sequence_lengths = np.sum(mask, axis=1).astype(np.int32)
        pred = []
        for utt, l in zip(pred_idxs, sequence_lengths):
            pred.append(utt[:l])
        return pred

    def predict_crf(self, xs):
        feed_dict = self._fill_feed_dict(xs)
        logits, trans_params, mask = self.sess.run([self._logits,
                                                    self._transition_params,
                                                    self.mask_ph],
                                                   feed_dict=feed_dict)
        sequence_lengths = np.maximum(np.sum(mask, axis=1).astype(np.int32), 1)
        # iterate over the sentences because no batching in viterbi_decode
        y_pred = []
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:int(sequence_length)]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            y_pred += [viterbi_seq]
        return y_pred

    def _fill_feed_dict(self, xs, y=None, learning_rate=None, train=False):
        assert len(xs) == len(self._xs_ph_list)
        xs = list(xs)
        xs[0] = np.array(xs[0])
        feed_dict = {ph: x for ph, x in zip(self._xs_ph_list, xs)}
        if y is not None:
            feed_dict[self._y_ph] = y
        if learning_rate is not None:
            feed_dict[self.learning_rate_ph] = learning_rate
        feed_dict[self.training_ph] = train
        if not train:
            feed_dict[self._dropout_ph] = 1.0
        return feed_dict

    def __call__(self, *args, **kwargs):
        if len(args[0]) == 0 or (len(args[0]) == 1 and len(args[0][0]) == 0):
            return []
        return self.predict(args)

    def train_on_batch(self, *args):
        *xs, y = args
        feed_dict = self._fill_feed_dict(xs, y, train=True, learning_rate=self._learning_rate)
        self.sess.run(self.train_op, feed_dict)

    def process_event(self, event_name, data):
        if event_name == 'after_validation':
            if not hasattr(self, '_best_f1'):
                self._best_f1 = 0
            if not hasattr(self, '_impatience'):
                self._impatience = 0
            if data['metrics']['ner_f1'] > self._best_f1:
                self._best_f1 = data['metrics']['ner_f1']
                self._impatience = 0
            else:
                self._impatience += 1

            if self._impatience >= self._lr_drop_patience:
                self._impatience = 0
                log.info('Dropping learning rate from {:.1e} to {:.1e}'.format(self._learning_rate,
                                                                               self._learning_rate * self._lr_drop_value))
                self.load()
                self._learning_rate *= self._lr_drop_value
