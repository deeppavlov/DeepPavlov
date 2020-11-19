# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import string
from logging import getLogger

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import LRScheduledTFModel

log = getLogger(__name__)


@register('hybrid_ner_model')
class HybridNerModel(LRScheduledTFModel):
    """ This class implements the hybrid NER model published in the paper: http://www.ijmlc.org/show-83-881-1.html

    Params:
        n_tags: Number of pre-defined tags.
        word_emb_path: The path to the pretrained word embedding model.
        word_emb_name: The name of pretrained word embedding model.
            One of the two values should be set including 'glove', 'baomoi' corresponding to two pre-trained word
            embedding models: GloVe (https://www.aclweb.org/anthology/D14-1162/)
            and baomoi (https://github.com/sonvx/word2vecVN). Otherwise, the word lookup table will be trained
            from scratch.
        word_vocab: The word vocabulary class.
        word_dim: The dimension of the pretrained word vector.
        char_vocab_size: The size of character vocabulary.
        pos_vocab_size: The size of POS vocabulary.
        chunk_vocab_size: The size of Chunk vocabulary.
        char_dim: The dimension of character vector.
        elmo_dim: The dimension of ELMo-based word vector
        elmo_hub_path: The path to the ELmo tensorhub
        pos_dim: The dimension of POS vector.
        chunk_dim: The dimension of Chunk vector.
        cap_dim: The dimension of capitalization vector.
        cap_vocab_size: The size of capitalization vocabulary.
        lstm_hidden_size: The number of units in contextualized Bi-LSTM network
        drop_out_keep_prob: The probability of keeping hidden state
    """

    def __init__(self,
                 n_tags: int,
                 word_vocab,
                 word_dim: int,
                 word_emb_path: str,
                 word_emb_name: str = None,
                 char_vocab_size: int = None,
                 pos_vocab_size: int = None,
                 chunk_vocab_size: int = None,
                 char_dim: int = None,
                 elmo_dim: int = None,
                 elmo_hub_path: str = "https://tfhub.dev/google/elmo/2",
                 pos_dim: int = None,
                 chunk_dim: int = None,
                 cap_dim: int = None,
                 cap_vocab_size: int = 5,
                 lstm_hidden_size: int = 256,
                 dropout_keep_prob: float = 0.5,
                 **kwargs) -> None:

        assert n_tags != 0, 'Number of classes equal 0! It seems that vocabularies is not loaded.' \
                            ' Check that all vocabulary files are downloaded!'

        if 'learning_rate_drop_div' not in kwargs:
            kwargs['learning_rate_drop_div'] = 10.0
        if 'learning_rate_drop_patience' not in kwargs:
            kwargs['learning_rate_drop_patience'] = 5.0
        if 'clip_norm' not in kwargs:
            kwargs['clip_norm'] = 5.0
        super().__init__(**kwargs)

        word2id = word_vocab.t2i
        word_emb_path = str(expand_path(word_emb_path))

        self._dropout_ph = tf.placeholder_with_default(dropout_keep_prob, shape=[], name='dropout')
        self.training_ph = tf.placeholder_with_default(False, shape=[], name='is_training')
        self._y_ph = tf.placeholder(tf.int32, [None, None], name='y_ph')

        self._xs_ph_list = []
        self._input_features = []

        # use for word contextualized bi-lstm, elmo
        self.real_sent_lengths_ph = tf.placeholder(tf.int32, [None], name="real_sent_lengths")
        self._xs_ph_list.append(self.real_sent_lengths_ph)

        # Word emb
        with tf.variable_scope("word_emb"):
            word_ids_ph = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self._xs_ph_list.append(word_ids_ph)

            word_embeddings = self.load_pretrained_word_emb(word_emb_path, word_emb_name, word_dim, word2id)

            word_lookup_table = tf.Variable(word_embeddings, dtype=tf.float32, trainable=True, name="word_embeddings")
            word_emb = tf.nn.embedding_lookup(word_lookup_table, word_ids_ph, name="embedded_word")
            self._input_features.append(word_emb)

        # POS feature
        if pos_dim is not None:
            with tf.variable_scope("pos_emb"):
                pos_ph = tf.placeholder(tf.int32, [None, None], name="pos_ids")
                self._xs_ph_list.append(pos_ph)

                tf_pos_embeddings = tf.get_variable(name="pos_embeddings",
                                                    dtype=tf.float32,
                                                    shape=[pos_vocab_size, pos_dim],
                                                    trainable=True,
                                                    initializer=xavier_initializer())

                embedded_pos = tf.nn.embedding_lookup(tf_pos_embeddings,
                                                      pos_ph,
                                                      name="embedded_pos")
                self._input_features.append(embedded_pos)

        # Chunk feature
        if chunk_dim is not None:
            with tf.variable_scope("chunk_emb"):
                chunk_ph = tf.placeholder(tf.int32, [None, None], name="chunk_ids")
                self._xs_ph_list.append(chunk_ph)

                tf_chunk_embeddings = tf.get_variable(name="chunk_embeddings",
                                                      dtype=tf.float32,
                                                      shape=[chunk_vocab_size, chunk_dim],
                                                      trainable=True,
                                                      initializer=xavier_initializer())

                embedded_chunk = tf.nn.embedding_lookup(tf_chunk_embeddings,
                                                        chunk_ph,
                                                        name="embedded_chunk")
                self._input_features.append(embedded_chunk)

        # Capitalization feature
        if cap_dim is not None:
            with tf.variable_scope("cap_emb"):
                cap_ph = tf.placeholder(tf.int32, [None, None], name="cap_ids")
                self._xs_ph_list.append(cap_ph)

                tf_cap_embeddings = tf.get_variable(name="cap_embeddings",
                                                    dtype=tf.float32,
                                                    shape=[cap_vocab_size, cap_dim],
                                                    trainable=True,
                                                    initializer=xavier_initializer())

                embedded_cap = tf.nn.embedding_lookup(tf_cap_embeddings,
                                                      cap_ph,
                                                      name="embedded_cap")
                self._input_features.append(embedded_cap)

        # Character feature
        if char_dim is not None:
            with tf.variable_scope("char_emb"):
                char_ids_ph = tf.placeholder(tf.int32, [None, None, None], name="char_ids")
                self._xs_ph_list.append(char_ids_ph)

                tf_char_embeddings = tf.get_variable(name="char_embeddings",
                                                     dtype=tf.float32,
                                                     shape=[char_vocab_size, char_dim],
                                                     trainable=True,
                                                     initializer=xavier_initializer())
                embedded_cnn_chars = tf.nn.embedding_lookup(tf_char_embeddings,
                                                            char_ids_ph,
                                                            name="embedded_cnn_chars")
                conv1 = tf.layers.conv2d(inputs=embedded_cnn_chars,
                                         filters=128,
                                         kernel_size=(1, 3),
                                         strides=(1, 1),
                                         padding="same",
                                         name="conv1",
                                         kernel_initializer=xavier_initializer_conv2d())
                conv2 = tf.layers.conv2d(inputs=conv1,
                                         filters=128,
                                         kernel_size=(1, 3),
                                         strides=(1, 1),
                                         padding="same",
                                         name="conv2",
                                         kernel_initializer=xavier_initializer_conv2d())
                char_cnn = tf.reduce_max(conv2, axis=2)

                self._input_features.append(char_cnn)

        # ELMo
        if elmo_dim is not None:
            with tf.variable_scope("elmo_emb"):
                padded_x_tokens_ph = tf.placeholder(tf.string, [None, None], name="padded_x_tokens")
                self._xs_ph_list.append(padded_x_tokens_ph)

                elmo = hub.Module(elmo_hub_path, trainable=True)
                emb = elmo(inputs={"tokens": padded_x_tokens_ph, "sequence_len": self.real_sent_lengths_ph},
                           signature="tokens", as_dict=True)["elmo"]
                elmo_emb = tf.layers.dense(emb, elmo_dim, activation=None)
                self._input_features.append(elmo_emb)

        features = tf.nn.dropout(tf.concat(self._input_features, axis=2), self._dropout_ph)

        with tf.variable_scope("bi_lstm_words"):
            cell_fw = tf.contrib.rnn.LSTMCell(lstm_hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(lstm_hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, features,
                                                                        sequence_length=self.real_sent_lengths_ph,
                                                                        dtype=tf.float32)
            self.output = tf.concat([output_fw, output_bw], axis=-1)

            ntime_steps = tf.shape(self.output)[1]
            self.output = tf.reshape(self.output, [-1, 2 * lstm_hidden_size])
            layer1 = tf.nn.dropout(tf.layers.dense(inputs=self.output, units=lstm_hidden_size, activation=None,
                                                   kernel_initializer=xavier_initializer()), self._dropout_ph)
            pred = tf.layers.dense(inputs=layer1, units=n_tags, activation=None,
                                   kernel_initializer=xavier_initializer())
            self.logits = tf.reshape(pred, [-1, ntime_steps, n_tags])

            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits,
                                                                                       self._y_ph,
                                                                                       self.real_sent_lengths_ph)
        # loss and opt
        with tf.variable_scope("loss_and_opt"):
            self.loss = tf.reduce_mean(-log_likelihood)
            self.train_op = self.get_train_op(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.load()

    def predict(self, xs):
        feed_dict = self._fill_feed_dict(xs)
        logits, trans_params, sent_lengths = self.sess.run([self.logits,
                                                            self.transition_params,
                                                            self.real_sent_lengths_ph],
                                                           feed_dict=feed_dict)
        # iterate over the sentences because no batching in viterbi_decode
        y_pred = []
        for logit, sequence_length in zip(logits, sent_lengths):
            logit = logit[:int(sequence_length)]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            y_pred += [viterbi_seq]
        return y_pred

    def _fill_feed_dict(self, xs, y=None, train=False):
        assert len(xs) == len(self._xs_ph_list)
        xs = list(xs)
        for x in xs[1:]:
            x = np.array(x)
        feed_dict = {ph: x for ph, x in zip(self._xs_ph_list, xs)}
        if y is not None:
            feed_dict[self._y_ph] = y
        feed_dict[self.training_ph] = train
        if not train:
            feed_dict[self._dropout_ph] = 1.0

        return feed_dict

    def __call__(self, *args, **kwargs):
        if len(args[0]) == 0 or (args[0] == [0]):
            return []
        return self.predict(args)

    def train_on_batch(self, *args):
        *xs, y = args
        feed_dict = self._fill_feed_dict(xs, y, train=True)
        _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict)
        return {'loss': loss_value,
                'learning_rate': self.get_learning_rate(),
                'momentum': self.get_momentum()}

    def load_pretrained_word_emb(self, model_path, model_name, word_dim, word2id=None, vocab_size=None):
        loaded_words = 0
        if word2id is not None:
            vocab_size = len(word2id)
        word_embeddings = np.zeros(shape=(vocab_size, word_dim))

        if model_name == "glove":
            model = KeyedVectors.load_word2vec_format(model_path, binary=False)
            for word in word2id:
                if word in model:
                    word_embeddings[word2id[word]] = model[word]
                    loaded_words += 1
        elif model_name == "baomoi":
            model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
            for word in word2id:
                if len(word) == 1:
                    if word[0] in string.punctuation:
                        word_embeddings[word2id[word]] = model["<punct>"]
                        loaded_words += 1
                elif word.isdigit():
                    word_embeddings[word2id[word]] = model["<number>"]
                    loaded_words += 1
                elif word in model.vocab:
                    word_embeddings[word2id[word]] = model[word]
                    loaded_words += 1
        elif model_name == "fasttext":
            ft_model = FastText.load_fasttext_format(model_path)
            for word in word2id:
                if word in ft_model.wv.vocab:
                    word_embeddings[word2id[word]] = ft_model.wv[word]
                    loaded_words += 1
        elif model_name is not None:
            raise RuntimeError(f'got an unexpected value for model_name: `{model_name}`')

        log.info(f"{loaded_words}/{vocab_size} words were loaded from {model_path}.")
        return word_embeddings
