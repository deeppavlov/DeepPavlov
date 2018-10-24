"""
Copyright 2018 Neural Networks and Deep Learning lab, MIPT

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
from typing import List, Iterable, Union
import tensorflow as tf
import tensorflow_hub as hub

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.tf_model import TFModel

log = get_logger(__name__)


class TensorflowBaseMatchingModel(TFModel):
    """
    Base class for ranking models that uses context-response matching schemes.

    Note:
        Tensorflow session variable already presents as self.sess attribute
        (derived from TFModel and initialized by Chainer)

    Args:
        batch_size (int): a number of samples in a batch

    """

    def __init__(self,
                 batch_size: int,
                 *args,
                 **kwargs):
        self.batch_size = batch_size

        g_1 = tf.Graph()
        with g_1.as_default():
            with tf.device('/cpu:0'):
                # sentence encoder
                self.embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False)

                # Raw sentences for context and response
                self.context_sent_ph = tf.placeholder(tf.string, shape=(None, self.num_context_turns))
                self.response_sent_ph = tf.placeholder(tf.string, shape=(None,))
                # embed sentences of context
                with tf.variable_scope('sentence_embeddings'):
                    x = []
                    for i in range(self.num_context_turns):
                        x.append(self.embed(tf.squeeze(self.context_sent_ph[:, i])))
                    embed_context_turns = tf.stack(x, axis=1)
                    embed_response = self.embed(self.response_sent_ph)

                    embed_context_turns = tf.reshape(embed_context_turns, shape=(-1, self.num_context_turns, 512))

                    dense_emb = tf.layers.Dense(200, kernel_initializer=tf.glorot_uniform_initializer(seed=42), trainable=False)

                    a = []
                    for i in range(self.num_context_turns):
                        a.append(dense_emb(embed_context_turns[:, i]))
                    self.sent_embedder_context = tf.stack(a, axis=1)
                    self.sent_embedder_response = dense_emb(embed_response)

                    self.sent_embedder_context = tf.expand_dims(self.sent_embedder_context, axis=2)
                    self.sent_embedder_response = tf.expand_dims(self.sent_embedder_response, axis=1)

                    # self.sent_embedder_context = tf.constant(1, dtype=tf.float32)
                    # self.sent_embedder_response = tf.constant(2, dtype=tf.float32)

            self.cpu_sess = tf.Session(config=tf.ConfigProto())
            self.cpu_sess.run(tf.global_variables_initializer())

        super(TensorflowBaseMatchingModel, self).__init__(*args, **kwargs)

    def __call__(self, samples_generator: Iterable[List[np.ndarray]]) -> Union[np.ndarray, List[str]]:
        y_pred = []
        batch_buffer_context = []       # [batch_size, 10, 50]
        raw_batch_buffer_context = []       # [batch_size, 10]
        batch_buffer_context_len = []   # [batch_size, 10]
        batch_buffer_response = []      # [batch_size, 50]
        raw_batch_buffer_response = []      # [batch_size]
        batch_buffer_response_len = []  # [batch_size]
        j = 0
        while True:
            try:
                sample = next(samples_generator)
                j += 1
                context_sentences = sample[:self.num_context_turns]
                response_sentences = sample[self.num_context_turns:self.num_context_turns*2]

                raw_context_sentences = sample[self.num_context_turns*2: self.num_context_turns*3]
                raw_response_sentences = sample[self.num_context_turns*3: ]

                # format model inputs
                # word indices
                batch_buffer_context += [context_sentences for sent in response_sentences]
                batch_buffer_response += [response_sentence for response_sentence in response_sentences]

                raw_batch_buffer_context += [raw_context_sentences for sent in raw_response_sentences]
                raw_batch_buffer_response += [raw_sent for raw_sent in raw_response_sentences]

                # lens of sentences
                lens = []
                for context in [context_sentences for sent in response_sentences]:
                    context_sentences_lens = []
                    for sent in context:
                        context_sentences_lens.append(len(sent[sent != 0]) + 1)
                    lens.append(context_sentences_lens)
                batch_buffer_context_len += lens

                lens = []
                for context in [response_sentence for response_sentence in response_sentences]:
                    lens.append(len(context[context != 0]) + 1)
                batch_buffer_response_len += lens

                if len(batch_buffer_context) >= self.batch_size:
                    for i in range(len(batch_buffer_context) // self.batch_size):

                        # CPU Graph
                        sent_feed_dict = {
                            self.context_sent_ph: np.array(
                                raw_batch_buffer_context[i * self.batch_size: (i + 1) * self.batch_size]),
                            self.response_sent_ph: np.array(
                                raw_batch_buffer_response[i * self.batch_size: (i + 1) * self.batch_size])
                        }
                        a, b = self.cpu_sess.run([self.sent_embedder_context, self.sent_embedder_response],
                                                 feed_dict=sent_feed_dict)
                        print(a, b)

                        # GPU Graph
                        feed_dict = {
                            self.utterance_ph: np.array(batch_buffer_context[i*self.batch_size:(i+1)*self.batch_size]),
                            self.all_utterance_len_ph: np.array(batch_buffer_context_len[i*self.batch_size:(i+1)*self.batch_size]),
                            self.response_ph: np.array(batch_buffer_response[i*self.batch_size:(i+1)*self.batch_size]),
                            self.response_len_ph: np.array(batch_buffer_response_len[i*self.batch_size:(i+1)*self.batch_size]),


                        }
                        yp = self.sess.run(self.y_pred, feed_dict=feed_dict)
                        print(yp)
                        y_pred += list(yp[:, 1])
                    lenb = len(batch_buffer_context) % self.batch_size
                    if lenb != 0:
                        batch_buffer_context = batch_buffer_context[-lenb:]
                        batch_buffer_context_len = batch_buffer_context_len[-lenb:]
                        batch_buffer_response = batch_buffer_response[-lenb:]
                        batch_buffer_response_len = batch_buffer_response_len[-lenb:]

                        raw_batch_buffer_context = raw_batch_buffer_context[-lenb:]
                        raw_batch_buffer_response = raw_batch_buffer_response[-lenb:]
                    else:
                        batch_buffer_context = []
                        batch_buffer_context_len = []
                        batch_buffer_response = []
                        batch_buffer_response_len = []

                        raw_batch_buffer_context = []
                        raw_batch_buffer_response = []
            except StopIteration:
                if j == 1:
                    return ["Error! It is not intended to use the model in the interact mode."]
                if len(batch_buffer_context) != 0:
                    feed_dict = {
                        self.utterance_ph: np.array(
                            batch_buffer_context[i * self.batch_size: (i + 1) * self.batch_size]),
                        self.all_utterance_len_ph: np.array(
                            batch_buffer_context_len[i * self.batch_size: (i + 1) * self.batch_size]),
                        self.response_ph: np.array(
                            batch_buffer_response[i * self.batch_size: (i + 1) * self.batch_size]),
                        self.response_len_ph: np.array(
                            batch_buffer_response_len[i * self.batch_size: (i + 1) * self.batch_size]),

                        self.context_sent_ph: np.array(
                            raw_batch_buffer_context[i * self.batch_size: (i + 1) * self.batch_size]),
                        self.response_sent_ph: np.array(
                            raw_batch_buffer_response[i * self.batch_size: (i + 1) * self.batch_size])
                    }
                    yp = self.sess.run(self.y_pred, feed_dict=feed_dict)
                    y_pred += list(yp[:, 1])
                break
        y_pred = np.asarray(y_pred)
        if len(response_sentences) > 1:
            y_pred = np.reshape(y_pred, (j, len(response_sentences)))  # reshape to [batch_size, 10]
        return y_pred

    # load() and save() are inherited from TFModel

    def train_on_batch(self, x: List[np.ndarray], y: List[int]) -> float:
        """
        This method is called by trainer to make one training step on one batch.

        :param x: generator that returns
                  list of ndarray - words of all sentences represented as integers,
                  with shape: (number_of_context_turns + 1, max_number_of_words_in_a_sentence)
        :param y: tuple of labels, with shape: (batch_size, )
        :return: value of loss function on batch
        """
        batch_buffer_context = []       # [batch_size, 10, 50]
        batch_buffer_context_len = []   # [batch_size, 10]
        batch_buffer_response = []      # [batch_size, 50]
        batch_buffer_response_len = []  # [batch_size]
        j = 0
        while True:
            try:
                sample = next(x)
                j += 1
                context_sentences = sample[:self.num_context_turns]
                response_sentences = sample[self.num_context_turns:]

                # format model inputs
                # word indices
                batch_buffer_context += [context_sentences for sent in response_sentences]
                batch_buffer_response += [response_sentence for response_sentence in response_sentences]
                # lens of sentences
                lens = []
                for context in [context_sentences for sent in response_sentences]:
                    context_sentences_lens = []
                    for sent in context:
                        context_sentences_lens.append(len(sent[sent != 0]))
                    lens.append(context_sentences_lens)
                batch_buffer_context_len += lens

                lens = []
                for context in [response_sentence for response_sentence in response_sentences]:
                    lens.append(len(context[context != 0]))
                batch_buffer_response_len += lens

                if len(batch_buffer_context) >= self.batch_size:
                    for i in range(len(batch_buffer_context) // self.batch_size):
                        feed_dict = {
                            self.utterance_ph: np.array(
                                batch_buffer_context[i * self.batch_size:(i + 1) * self.batch_size]),
                            self.all_utterance_len_ph: np.array(
                                batch_buffer_context_len[i * self.batch_size:(i + 1) * self.batch_size]),
                            self.response_ph: np.array(
                                batch_buffer_response[i * self.batch_size:(i + 1) * self.batch_size]),
                            self.response_len_ph: np.array(
                                batch_buffer_response_len[i * self.batch_size:(i + 1) * self.batch_size]),
                            self.y_true: np.array(y)
                        }
                        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                    lenb = len(batch_buffer_context) % self.batch_size
                    if lenb != 0:
                        batch_buffer_context = batch_buffer_context[-lenb:]
                        batch_buffer_context_len = batch_buffer_context_len[-lenb:]
                        batch_buffer_response = batch_buffer_response[-lenb:]
                        batch_buffer_response_len = batch_buffer_response_len[-lenb:]
                    else:
                        batch_buffer_context = []
                        batch_buffer_context_len = []
                        batch_buffer_response = []
                        batch_buffer_response_len = []
            except StopIteration:
                if j == 1:
                    return ["Error! It is not intended to use the model in the interact mode."]
                if len(batch_buffer_context) != 0:
                    feed_dict = {
                        self.utterance_ph: np.array(
                            batch_buffer_context[i * self.batch_size:(i + 1) * self.batch_size]),
                        self.all_utterance_len_ph: np.array(
                            batch_buffer_context_len[i * self.batch_size:(i + 1) * self.batch_size]),
                        self.response_ph: np.array(
                            batch_buffer_response[i * self.batch_size:(i + 1) * self.batch_size]),
                        self.response_len_ph: np.array(
                            batch_buffer_response_len[i * self.batch_size:(i + 1) * self.batch_size]),
                        self.y_true: np.array(y)
                    }
                    loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                break
        return loss
