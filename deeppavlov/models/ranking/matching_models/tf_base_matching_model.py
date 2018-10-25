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
        super(TensorflowBaseMatchingModel, self).__init__(*args, **kwargs)

    def __call__(self, samples_generator: Iterable[List[np.ndarray]]) -> Union[np.ndarray, List[str]]:
        y_pred = []
        batch_buffer_context = []       # [batch_size, 10, 50]
        batch_buffer_context_len = []   # [batch_size, 10]
        batch_buffer_response = []      # [batch_size, 50]
        batch_buffer_response_len = []  # [batch_size]

        raw_batch_buffer_context = []  # [batch_size, 10]
        raw_batch_buffer_response = []  # [batch_size]
        j = 0
        while True:
            try:
                sample = next(samples_generator)
                sample_len = len(sample)
                j += 1
                context_sentences = sample[:self.num_context_turns]
                response_sentences = sample[self.num_context_turns:sample_len//2]

                raw_context_sentences = sample[sample_len//2:sample_len//2 + self.num_context_turns]
                raw_response_sentences = sample[sample_len//2 + self.num_context_turns:]

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
                        sent_len = len(sent[sent != 0])
                        sent_len = sent_len + 1 if sent_len > 0 else 0
                        context_sentences_lens.append(sent_len)
                    lens.append(context_sentences_lens)
                batch_buffer_context_len += lens

                lens = []
                for context in [response_sentence for response_sentence in response_sentences]:
                    sent_len = len(context[context != 0])
                    sent_len = sent_len + 1 if sent_len > 0 else 0
                    lens.append(sent_len)
                batch_buffer_response_len += lens

                if len(batch_buffer_context) >= self.batch_size:
                    for i in range(len(batch_buffer_context) // self.batch_size):

                        # CPU Graph
                        with self.g_cpu.as_default():
                            sent_feed_dict = {
                                self.context_sent_ph: np.array(
                                    raw_batch_buffer_context[i * self.batch_size: (i + 1) * self.batch_size]),
                                self.response_sent_ph: np.array(
                                    raw_batch_buffer_response[i * self.batch_size: (i + 1) * self.batch_size])
                            }
                            c, r = self.cpu_sess.run([self.sent_embedder_context, self.sent_embedder_response],
                                                     feed_dict=sent_feed_dict)

                        # GPU Graph
                        with self.graph.as_default():
                            feed_dict = {
                                self.utterance_ph: np.array(
                                    batch_buffer_context[i * self.batch_size:(i + 1) * self.batch_size]),
                                self.all_utterance_len_ph: np.array(
                                    batch_buffer_context_len[i * self.batch_size:(i + 1) * self.batch_size]),
                                self.response_ph: np.array(
                                    batch_buffer_response[i * self.batch_size:(i + 1) * self.batch_size]),
                                self.response_len_ph: np.array(
                                    batch_buffer_response_len[i * self.batch_size:(i + 1) * self.batch_size]),

                                self.context_sent_emb_ph: c,
                                self.response_sent_emb_ph: r
                            }
                            yp = self.sess.run(self.y_pred, feed_dict=feed_dict)
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
                    # CPU Graph
                    with self.g_cpu.as_default():
                        sent_feed_dict = {
                            self.context_sent_ph: np.array(raw_batch_buffer_context),
                            self.response_sent_ph: np.array(raw_batch_buffer_response)
                        }
                        c, r = self.cpu_sess.run([self.sent_embedder_context, self.sent_embedder_response],
                                                 feed_dict=sent_feed_dict)

                    # GPU Graph
                    with self.graph.as_default():
                        feed_dict = {
                            self.utterance_ph: np.array(batch_buffer_context),
                            self.all_utterance_len_ph: np.array(batch_buffer_context_len),
                            self.response_ph: np.array(batch_buffer_response),
                            self.response_len_ph: np.array(batch_buffer_response_len),

                            self.context_sent_emb_ph: c,
                            self.response_sent_emb_ph: r
                        }
                        yp = self.gpu_sess.run(self.y_pred, feed_dict=feed_dict)
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

        raw_batch_buffer_context = []  # [batch_size, 10]
        raw_batch_buffer_response = []  # [batch_size]
        j = 0
        while True:
            try:
                sample = next(x)
                sample_len = len(sample)
                j += 1
                context_sentences = sample[:self.num_context_turns]
                response_sentences = sample[self.num_context_turns:sample_len//2]

                raw_context_sentences = sample[sample_len//2:sample_len//2 + self.num_context_turns]
                raw_response_sentences = sample[sample_len//2 + self.num_context_turns:]

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
                        sent_len = len(sent[sent != 0])
                        sent_len = sent_len + 1 if sent_len > 0 else 0
                        context_sentences_lens.append(sent_len)
                    lens.append(context_sentences_lens)
                batch_buffer_context_len += lens

                lens = []
                for context in [response_sentence for response_sentence in response_sentences]:
                    sent_len = len(context[context != 0])
                    sent_len = sent_len + 1 if sent_len > 0 else 0
                    lens.append(sent_len)
                batch_buffer_response_len += lens

                if len(batch_buffer_context) >= self.batch_size:
                    for i in range(len(batch_buffer_context) // self.batch_size):
                        # CPU Graph
                        with self.g_cpu.as_default():
                            sent_feed_dict = {
                                self.context_sent_ph: np.array(
                                    raw_batch_buffer_context[i * self.batch_size: (i + 1) * self.batch_size]),
                                self.response_sent_ph: np.array(
                                    raw_batch_buffer_response[i * self.batch_size: (i + 1) * self.batch_size])
                            }
                            c, r = self.cpu_sess.run([self.sent_embedder_context, self.sent_embedder_response],
                                                     feed_dict=sent_feed_dict)

                        # GPU Graph
                        with self.graph.as_default():
                            feed_dict = {
                                self.utterance_ph: np.array(
                                    batch_buffer_context[i * self.batch_size:(i + 1) * self.batch_size]),
                                self.all_utterance_len_ph: np.array(
                                    batch_buffer_context_len[i * self.batch_size:(i + 1) * self.batch_size]),
                                self.response_ph: np.array(
                                    batch_buffer_response[i * self.batch_size:(i + 1) * self.batch_size]),
                                self.response_len_ph: np.array(
                                    batch_buffer_response_len[i * self.batch_size:(i + 1) * self.batch_size]),
                                self.y_true: np.array(y),

                                self.context_sent_emb_ph: c,
                                self.response_sent_emb_ph: r
                            }
                            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
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
                    # CPU Graph
                    with self.g_cpu.as_default():
                        sent_feed_dict = {
                            self.context_sent_ph: np.array(raw_batch_buffer_context),
                            self.response_sent_ph: np.array(raw_batch_buffer_response)
                        }
                        c, r = self.cpu_sess.run([self.sent_embedder_context, self.sent_embedder_response],
                                                 feed_dict=sent_feed_dict)
                    # GPU Graph
                    with self.graph.as_default():
                        feed_dict = {
                            self.utterance_ph: np.array(batch_buffer_context),
                            self.all_utterance_len_ph: np.array(batch_buffer_context_len),
                            self.response_ph: np.array(batch_buffer_response),
                            self.response_len_ph: np.array(batch_buffer_response_len),
                            self.y_true: np.array(y),

                            self.context_sent_emb_ph: c,
                            self.response_sent_emb_ph: r
                        }
                        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                break
        return loss
