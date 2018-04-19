"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import tensorflow as tf

from deeppavlov.core.common.utils import check_gpu_existance
from tensorflow.contrib.layers import xavier_initializer
from nltk.tag import SennaNERTagger, SennaChunkTagger


class NerNetwork:
    def __init__(self,
                 embedder,
                 tag_vocab,
                 ner_vocab,
                 pos_vocab,
                 sess=None):

        # check gpu
        if not check_gpu_existance():
            raise RuntimeError('Ontonotes NER model requires GPU with cuDNN!')

        n_hidden=(256, 256, 256)
        token_embeddings_dim=100
        n_tags = len(tag_vocab)

        # Create placeholders
        x_word = tf.placeholder(dtype=tf.float32, shape=[None, None, token_embeddings_dim], name='x_word')
        x_char = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')

        # Features
        x_pos = tf.placeholder(dtype=tf.float32, shape=[None, None, len(pos_vocab)], name='x_pos')  # Senna
        x_ner = tf.placeholder(dtype=tf.float32, shape=[None, None, len(ner_vocab)], name='x_ner')  # Senna
        x_capi = tf.placeholder(dtype=tf.float32, shape=[None, None], name='x_capi')

        y_true = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')
        mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name='mask')
        sequence_lengths = tf.reduce_sum(mask, axis=1)

        # Concat features to embeddings
        emb = tf.concat([x_word, tf.expand_dims(x_capi, 2), x_pos, x_ner], axis=2)

        # The network
        units = emb
        for n, n_h in enumerate(n_hidden):
            with tf.variable_scope('RNN_' + str(n)):
                units, _ = cudnn_bi_lstm(units, n_h, tf.to_int32(sequence_lengths))

        # Classifier
        with tf.variable_scope('Classifier'):
            units = tf.layers.dense(units, n_hidden[-1], kernel_initializer=xavier_initializer())
            logits = tf.layers.dense(units, n_tags, kernel_initializer=xavier_initializer())

        # CRF
        _, trainsition_params = tf.contrib.crf.crf_log_likelihood(logits,
                                                                  y_true,
                                                                  sequence_lengths)

        # Initialize session
        if sess is None:
            sess = tf.Session()

        self._ner_tagger = SennaNERTagger('download/senna/')
        self._pos_tagger = SennaChunkTagger('download/senna/')

        self._x_w = x_word
        self._x_c = x_char
        self._x_capi = x_capi
        self.x_pos = x_pos
        self.x_ner = x_ner
        self._y_true = y_true
        self._mask = mask
        self._sequence_lengths = sequence_lengths
        self._token_embeddings_dim = token_embeddings_dim

        self._pos_dict = pos_vocab
        self._ner_dict = ner_vocab
        self._tag_dict = tag_vocab

        self._logits = logits
        self._trainsition_params = trainsition_params

        self._sess = sess
        sess.run(tf.global_variables_initializer())
        self._embedder = embedder

    def load(self, model_file_path):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self._sess, model_file_path)

    @staticmethod
    def to_one_hot(x, n):
        b = np.zeros([len(x), n], dtype=np.float32)
        for q, tok in enumerate(x):
            b[q, tok] = 1
        return b

    def tokens_batch_to_numpy_batch(self, batch_x):
        """ Convert a batch of tokens to numpy arrays of features"""
        x = dict()
        batch_size = len(batch_x)
        max_utt_len = max([len(utt) for utt in batch_x])

        # Embeddings
        x['emb'] = self._embedder(batch_x)

        # Capitalization
        x['capitalization'] = np.zeros([batch_size, max_utt_len], dtype=np.float32)
        for n, utt in enumerate(batch_x):
            x['capitalization'][n, :len(utt)] = [tok[0].isupper() for tok in utt]

        # POS
        n_pos = len(self._pos_dict)
        x['pos'] = np.zeros([batch_size, max_utt_len, n_pos])
        for n, utt in enumerate(batch_x):
            token_tag_pairs = self._pos_tagger.tag(utt)
            pos_tags = list(zip(*token_tag_pairs))[1]
            pos = np.array([self._pos_dict[p] for p in pos_tags])
            pos = self.to_one_hot(pos, n_pos)
            x['pos'][n, :len(pos)] = pos

        # NER
        n_ner = len(self._ner_dict)
        x['ner'] = np.zeros([batch_size, max_utt_len, n_ner])
        for n, utt in enumerate(batch_x):
            token_tag_pairs = self._ner_tagger.tag(utt)
            ner_tags = list(zip(*token_tag_pairs))[1]
            ner = np.array([self._ner_dict[p] for p in ner_tags])
            ner = self.to_one_hot(ner, n_ner)
            x['ner'][n, :len(ner)] = ner

        # Mask for paddings
        x['mask'] = np.zeros([batch_size, max_utt_len], dtype=np.float32)
        for n in range(batch_size):
            x['mask'][n, :len(batch_x[n])] = 1

        return x

    def train_on_batch(self, x_word, x_char, y_tag):
        raise NotImplementedError

    def predict(self, x):
        feed_dict = self._fill_feed_dict(x)
        y_pred = []
        logits, trans_params, sequence_lengths = self._sess.run([self._logits,
                                                                 self._trainsition_params,
                                                                 self._sequence_lengths
                                                                 ],
                                                                feed_dict=feed_dict)

        # iterate over the sentences because no batching in viterbi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:int(sequence_length)]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            y_pred += [viterbi_seq]

        pred = []
        batch_size = len(x['emb'])
        for n in range(batch_size):
            pred.append([self._tag_dict[tag] for tag in y_pred[n]])
        return pred

    def predict_on_batch(self, tokens_batch):
        batch_x = self.tokens_batch_to_numpy_batch(tokens_batch)
        # Prediction indices
        predictions_batch = self.predict(batch_x)
        predictions_batch_no_pad = list()
        for n, predicted_tags in enumerate(predictions_batch):
            predictions_batch_no_pad.append(predicted_tags[: len(tokens_batch[n])])
        return predictions_batch_no_pad

    def _fill_feed_dict(self, x):

        feed_dict = dict()
        feed_dict[self._x_w] = x['emb']
        feed_dict[self._mask] = x['mask']

        feed_dict[self.x_pos] = x['pos']
        feed_dict[self.x_ner] = x['ner']

        feed_dict[self._x_capi] = x['capitalization']
        return feed_dict


def cudnn_bi_lstm(units, n_hidden, seq_lengths, n_layers=1):
    """ Fast CuDNN Bi-LSTM implementation

        Args:
            units: tf.Tensor with dimensions [B x T x F], where
                B - batch size
                T - number of tokens
                F - features
            n_hidden: dimensionality of hidden state
            n_layers: number of layers

        Returns:
            h - all hidden states along T dimension,
                tf.Tensor with dimensionality [B x T x F]
            h_last - last hidden state, tf.Tensor with dimensionality [B x H * 2]
                where H - number of hidden units
            c_last - last cell state, tf.Tensor with dimensionality [B x H * 2]
                where H - number of hidden units
        """
    lstm_fw = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=n_layers,
                                             num_units=n_hidden,
                                             input_size=units.get_shape().as_list()[-1])
    lstm_bw = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=n_layers,
                                             num_units=n_hidden,
                                             input_size=units.get_shape().as_list()[-1])

    param_fw = tf.Variable(tf.random_uniform(
        [lstm_fw.params_size()], -0.1, 0.1), validate_shape=False)
    param_bw = tf.Variable(tf.random_uniform(
        [lstm_bw.params_size()], -0.1, 0.1), validate_shape=False)

    init_h_fw = tf.zeros([1, tf.shape(units)[0], n_hidden])
    init_h_bw = tf.zeros([1, tf.shape(units)[0], n_hidden])

    init_c_fw = tf.zeros([1, tf.shape(units)[0], n_hidden])
    init_c_bw = tf.zeros([1, tf.shape(units)[0], n_hidden])

    h_fw, h_last_fw, c_last_fw = lstm_fw(tf.transpose(units, (1, 0, 2)), init_h_fw, init_c_fw, param_fw)
    reversed_units = tf.reverse_sequence(units, seq_lengths=seq_lengths, seq_dim=1, batch_dim=0)
    h_bw, h_last_bw, c_last_bw = lstm_bw(tf.transpose(reversed_units, (1, 0, 2)), init_h_bw, init_c_bw, param_bw)
    h_bw = tf.reverse_sequence(h_bw, seq_lengths=seq_lengths, seq_dim=0, batch_dim=1)

    h = tf.concat([h_fw, h_bw], axis=2)
    h_last = tf.concat([h_last_fw, h_last_bw], axis=2)
    c_last = tf.concat([c_last_fw, c_last_bw], axis=2)

    h = tf.transpose(h, (1, 0, 2))
    h_last = tf.squeeze(h_last, 0)
    return h, (h_last, c_last)