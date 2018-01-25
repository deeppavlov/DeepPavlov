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
from collections import defaultdict
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from deeppavlov.models.ner.layers import character_embedding_network
from deeppavlov.models.ner.layers import embedding_layer
from deeppavlov.models.ner.layers import highway_convolutional_network
from deeppavlov.models.ner.layers import stacked_convolutions
from deeppavlov.models.ner.layers import stacked_rnn
from deeppavlov.models.ner.evaluation import precision_recall_f1



SEED = 42
MODEL_FILE_NAME = 'ner_model'


class NerNetwork:
    def __init__(self,
                 word_vocab,
                 char_vocab,
                 tag_vocab,
                 n_filters=(128, 256),
                 filter_width=3,
                 token_embeddings_dim=128,
                 char_embeddings_dim=50,
                 use_char_embeddins=True,
                 embeddings_dropout=False,
                 dense_dropout=False,
                 use_batch_norm=False,
                 logging=False,
                 entity_of_interest=None,
                 use_crf=False,
                 net_type='cnn',
                 char_filter_width=5,
                 verbouse=False,
                 embeddings_onethego=False):
        n_tags = len(tag_vocab)
        n_tokens = len(word_vocab)
        n_chars = len(char_vocab)

        # Create placeholders
        if embeddings_onethego:
            x_word = tf.placeholder(dtype=tf.float32, shape=[None, None, token_embeddings_dim], name='x_word')
        else:
            x_word = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_word')
        x_char = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')
        y_true = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')

        # Auxiliary placeholders
        learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        dropout_ph = tf.placeholder_with_default(1.0, shape=[])
        training_ph = tf.placeholder_with_default(False, shape=[])
        mask_ph = tf.placeholder(dtype=tf.float32, shape=[None, None])

        # Embeddings
        if not embeddings_onethego:
            with tf.variable_scope('Embeddings'):
                w_emb = embedding_layer(x_word, n_tokens=n_tokens, token_embedding_dim=token_embeddings_dim)
                if use_char_embeddins:
                    c_emb = character_embedding_network(x_char,
                                                        n_characters=n_chars,
                                                        char_embedding_dim=char_embeddings_dim,
                                                        filter_width=char_filter_width)
                    emb = tf.concat([w_emb, c_emb], axis=-1)
                else:
                    emb = w_emb
        else:
            emb = x_word

        # Dropout for embeddings
        if embeddings_dropout:
            emb = tf.layers.dropout(emb, dropout_ph, training=training_ph)

        if 'cnn' in net_type.lower():
            # Convolutional network
            with tf.variable_scope('ConvNet'):
                units = stacked_convolutions(emb,
                                             n_filters=n_filters,
                                             filter_width=filter_width,
                                             use_batch_norm=use_batch_norm,
                                             training_ph=training_ph)
        elif 'rnn' in net_type.lower():
            units = stacked_rnn(emb, n_filters, cell_type='lstm')

        elif 'cnn_highway' in net_type.lower():
            units = highway_convolutional_network(emb,
                                                  n_filters=n_filters,
                                                  filter_width=filter_width,
                                                  use_batch_norm=use_batch_norm,
                                                  training_ph=training_ph)
        else:
            raise KeyError('There is no such type of network: {}'.format(net_type))
        # Classifier
        with tf.variable_scope('Classifier'):
            logits = tf.layers.dense(units, n_tags, kernel_initializer=xavier_initializer())

        # Loss with masking
        if use_crf:
            sequence_lengths = tf.reduce_sum(mask_ph, axis=1)
            log_likelihood, trainsition_params = tf.contrib.crf.crf_log_likelihood(logits,
                                                                                   y_true,
                                                                                   sequence_lengths)
            loss_tensor = -log_likelihood
            predictions = None
        else:
            ground_truth_labels = tf.one_hot(y_true, n_tags)
            loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_labels, logits=logits)
            loss_tensor = loss_tensor * mask_ph
            predictions = tf.argmax(logits, axis=-1)

        loss = tf.reduce_mean(loss_tensor)

        # Initialize session
        sess = tf.Session()
        if verbouse:
            self.print_number_of_parameters()
        if logging:
            self.train_writer = tf.summary.FileWriter('summary', sess.graph)

        self.token_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.char_vocab = char_vocab
        self._use_crf = use_crf
        self.summary = tf.summary.merge_all()
        self._x_w = x_word
        self._x_c = x_char
        self._y_true = y_true
        self._y_pred = predictions
        if use_crf:
            self._logits = logits
            self._trainsition_params = trainsition_params
            self._sequence_lengths = sequence_lengths
        self._loss = loss
        self._sess = sess
        self._learning_rate_ph = learning_rate_ph
        self._dropout = dropout_ph
        self._loss_tensor = loss_tensor
        self._use_dropout = True if embeddings_dropout or dense_dropout else None
        self._training_ph = training_ph
        self._logging = logging
        self._train_op = self.get_train_op(loss, learning_rate_ph)
        self._embeddings_onethego = embeddings_onethego
        self._entity_of_interest = entity_of_interest
        self.verbouse = verbouse
        self._mask = mask_ph
        sess.run(tf.global_variables_initializer())

    def tokens_batch_to_numpy_batch(self, batch_x, batch_y=None):
        # Determine dimensions
        batch_size = len(batch_x)
        max_utt_len = max([len(utt) for utt in batch_x])
        max_token_len = max([len(token) for utt in batch_x for token in utt])

        x_token = np.ones([batch_size, max_utt_len], dtype=np.int32) * self.token_vocab['<PAD>']
        x_char = np.ones([batch_size, max_utt_len, max_token_len], dtype=np.int32) * self.char_vocab['<PAD>']
        mask = np.zeros_like(x_token)
        if batch_y is not None:
            y = np.ones([batch_size, max_utt_len], dtype=np.int32) * self.tag_vocab['<PAD>']
        else:
            y = None

        # Prepare x batch
        for n, utterance in enumerate(batch_x):
            mask[n, :len(utterance)] = 1
            x_token[n, :len(utterance)] = self.token_vocab.toks2idxs(utterance)
            for k, token in enumerate(utterance):
                x_char[n, k, :len(token)] = self.char_vocab.toks2idxs(token)

        # Prepare y batch
        if batch_y is not None:
            for n, tags in enumerate(batch_y):
                y[n, :len(tags)] = self.tag_vocab.toks2idxs(tags)

        return (x_token, x_char, mask), y

    def eval_conll(self, data, print_results=True, short_report=True, data_type=None):
        y_true_list = []
        y_pred_list = []
        if data_type is not None:
            print('Eval on {}:'.format(data_type), file=sys.stderr)
        for x, y_gt in data:
            (x_token, x_char, mask), y = self.tokens_batch_to_numpy_batch([x])
            y_pred = self._predict(x_token, x_char, mask)
            y_pred = self.tag_vocab.batch_idxs2batch_toks(y_pred)
            for tags_pred, tags_gt in zip(y_pred, [y_gt]):
                for tag_predicted, tag_ground_truth in zip(tags_pred, tags_gt):
                    y_true_list.append(tag_ground_truth)
                    y_pred_list.append(tag_predicted)
                y_true_list.append('O')
                y_pred_list.append('O')
        return precision_recall_f1(y_true_list,
                                   y_pred_list,
                                   print_results,
                                   short_report)

    def train(self, data, batch_size=8, learning_rate=1e-3, dropout_rate=0.5):
        total_loss = 0
        total_count = 0
        for batch in data.batch_generator(batch_size):
            current_loss = self.train_on_batch(batch,
                                               learning_rate=learning_rate,
                                               dropout_rate=dropout_rate)
            total_loss += current_loss
            # Add len of x
            total_count += len(batch[0])

    def train_on_batch(self, batch, learning_rate=1e-3, dropout_rate=0.5):
        batch_x, batch_y = batch
        (x_toks, x_char, mask), y_tags = self.tokens_batch_to_numpy_batch(batch_x, batch_y)
        feed_dict = self._fill_feed_dict(x_toks,
                                         x_char,
                                         mask,
                                         y_tags,
                                         learning_rate,
                                         dropout_rate=dropout_rate,
                                         training=True)
        loss, _ = self._sess.run([self._loss, self._train_op], feed_dict=feed_dict)
        return loss

    def predict_on_batch(self, x_batch):
        (x_toks, x_char, mask), _ = self.tokens_batch_to_numpy_batch(x_batch)
        y_pred = self._predict(x_toks, x_char, mask)
        # TODO: add padding filtering
        y_pred_tags = self.tag_vocab.batch_idxs2batch_toks(y_pred)
        return y_pred_tags

    def _predict(self, x_word, x_char, mask=None):

        feed_dict = self._fill_feed_dict(x_word, x_char, mask, training=False)
        if self._use_crf:
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
        else:
            y_pred = self._sess.run(self._y_pred, feed_dict=feed_dict)
        return y_pred

    def fit(self, batch_gen=None, batch_size=32, learning_rate=1e-3, epochs=1, dropout_rate=0.5, learning_rate_decay=1):
        for epoch in range(epochs):
            count = 0
            if self.verbouse:
                print('Epoch {}'.format(epoch), file=sys.stderr)
            if batch_gen is None:
                batch_generator = self.corpus.batch_generator(batch_size, dataset_type='train')
            for (x_word, x_char), y_tag in batch_generator:

                feed_dict = self._fill_feed_dict(x_word,
                                                 x_char,
                                                 y_tag,
                                                 learning_rate,
                                                 dropout_rate=dropout_rate,
                                                 training=True)
                if self._logging:
                    summary, _ = self._sess.run([self.summary, self._train_op], feed_dict=feed_dict)
                    self.train_writer.add_summary(summary)

                self._sess.run(self._train_op, feed_dict=feed_dict)
                count += len(x_word)
            if self.verbouse:
                self.eval_conll('valid', print_results=True)
            self.save()

        if self.verbouse:
            self.eval_conll(dataset_type='train', short_report=False)
            self.eval_conll(dataset_type='valid', short_report=False)
            results = self.eval_conll(dataset_type='test', short_report=False)
        else:
            results = self.eval_conll(dataset_type='test', short_report=True)
        return results

    def infer(self, instance, *args, **kwargs):
        return self.predict_for_token_batch([instance])

    def _fill_feed_dict(self,
                        x_w,
                        x_c,
                        mask=None,
                        y_t=None,
                        learning_rate=None,
                        training=False,
                        dropout_rate=1):
        feed_dict = {}
        feed_dict[self._x_w] = x_w
        feed_dict[self._x_c] = x_c
        feed_dict[self._training_ph] = training
        if mask is not None:
            feed_dict[self._mask] = mask
        else:
            feed_dict[self._mask] = np.ones(x_w.shape[:2])
        if y_t is not None:
            feed_dict[self._y_true] = y_t
        if learning_rate is not None:
            feed_dict[self._learning_rate_ph] = learning_rate
        if self._use_dropout is not None and training:
            feed_dict[self._dropout] = dropout_rate
        else:
            feed_dict[self._dropout] = 1.0
        return feed_dict

    def eval_loss(self, data_generator):
        num_tokens = 0
        loss = 0
        for (x_w, x_c), y_t in data_generator:
            feed_dict = self._fill_feed_dict(x_w, x_c, y_t, training=False)
            loss += np.sum(self._sess.run(self._loss_tensor, feed_dict=feed_dict))
            num_tokens = np.sum(self._sess.run(self._sequence_lengths, feed_dict=feed_dict))
        return loss / num_tokens

    @staticmethod
    def get_trainable_variables(trainable_scope_names=None):
        vars = tf.trainable_variables()
        if trainable_scope_names is not None:
            vars_to_train = []
            for scope_name in trainable_scope_names:
                for var in vars:
                    if var.name.startswith(scope_name):
                        vars_to_train.append(var)
            return vars_to_train
        else:
            return vars

    def predict_for_token_batch(self, tokens_batch):
        # Determine dimensions
        batch_size = len(tokens_batch)
        max_utt_len = max([len(utt) for utt in tokens_batch])
        max_token_len = max([len(token) for utt in tokens_batch for token in utt])
        # Prepare numpy arrays
        x_token = np.ones([batch_size, max_utt_len], dtype=np.int32) * self.token_vocab['<PAD>']
        x_char = np.ones([batch_size, max_utt_len, max_token_len], dtype=np.int32) * self.char_vocab['<PAD>']
        mask = np.zeros([batch_size, max_utt_len], dtype=np.int32)
        # Prepare x batch
        for n, utterance in enumerate(tokens_batch):
            mask[n, :len(utterance)] = 1
            x_token[n, :len(utterance)] = self.token_vocab.toks2idxs(utterance)
            for k, token in enumerate(utterance):
                x_char[n, k, :len(token)] = self.char_vocab.toks2idxs(token)

        feed_dict = self._fill_feed_dict(x_token, x_char, mask)
        if self._use_crf:
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
        else:
            y_pred = self._sess.run(self._y_pred, feed_dict)

        return self.tag_vocab.batch_idxs2batch_toks(y_pred)

    def shutdown(self):
        self._sess.close()

    def save(self, model_file_path):
        """
        Save model to model_file_path
        """
        saver = tf.train.Saver()
        saver.save(self._sess, str(model_file_path))

    def load(self, model_file_path):
        """
        Load model from the model_file_path
        """
        saver = tf.train.Saver()
        saver.restore(self._sess, str(model_file_path))

    @staticmethod
    def print_number_of_parameters():
        """
        Print number of *trainable* parameters in the network
        """
        print('Number of parameters: ', file=sys.stderr)
        vars = tf.trainable_variables()
        blocks = defaultdict(int)
        for var in vars:
            # Get the top level scope name of variable
            block_name = var.name.split('/')[0]
            number_of_parameters = np.prod(var.get_shape().as_list())
            blocks[block_name] += number_of_parameters
        for block_name in blocks:
            print(block_name, blocks[block_name], file=sys.stderr)
        total_num_parameters = np.sum(list(blocks.values()))
        print('Total number of parameters equal {}'.format(total_num_parameters), file=sys.stderr)

    def get_train_op(self, loss, learning_rate, learnable_scopes=None, optimizer=None):
        """ Get train operation for given loss

        Args:
            loss: loss, tf tensor or scalar
            learning_rate: scalar or placeholder
            learnable_scopes: which scopes are trainable (None for all)
            optimizer: instance of tf.train.Optimizer, default Adam

        Returns:
            train_op
        """
        variables = self.get_trainable_variables(learnable_scopes)
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer

        # For batch norm it is necessary to update running averages
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer(learning_rate).minimize(loss, var_list=variables)
        return train_op
