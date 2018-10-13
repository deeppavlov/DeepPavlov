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

from deeppavlov.core.common.log import get_logger
from deeppavlov.models.ranking.siamese_model import SiameseModel
from deeppavlov.core.models.tf_model import TFModel

log = get_logger(__name__)


class TensorflowBaseMatchingModel(TFModel, SiameseModel):
    """
    Base class for ranking models that uses context-response matching schemes.

    Note:
        Tensorflow session variable already presents as self.sess attribute
        (derived from TFModel and initialized by Chainer)

    Args:
        none

    """

    def __init__(self,
                 *args,
                 **kwargs):
        super(TensorflowBaseMatchingModel, self).__init__(*args, **kwargs)

    def __call__(self, samples_generator: Iterable[List[np.ndarray]]) -> Union[np.ndarray, List[str]]:
        y_pred = []
        batch_buffer_context = []       # [batch_size, 10, 50]
        batch_buffer_context_len = []   # [batch_size, 10]
        batch_buffer_response = []      # [batch_size, 50]
        batch_buffer_response_len = []  # [batch_size]
        j = 0
        while True:
            try:
                sample = next(samples_generator)
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
                            self.utterance_ph: np.array(batch_buffer_context[i*self.batch_size:(i+1)*self.batch_size]),
                            self.all_utterance_len_ph: np.array(batch_buffer_context_len[i*self.batch_size:(i+1)*self.batch_size]),
                            self.response_ph: np.array(batch_buffer_response[i*self.batch_size:(i+1)*self.batch_size]),
                            self.response_len_ph: np.array(batch_buffer_response_len[i*self.batch_size:(i+1)*self.batch_size])
                        }
                        yp = self.sess.run(self.y_pred, feed_dict=feed_dict)
                        y_pred += list(yp[:, 1])
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
                            batch_buffer_response_len[i * self.batch_size:(i + 1) * self.batch_size])
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

        :param x: tuple of lists of ndarray - words of all sentences represented as integers,
                  with shape: (batch_size, number_of_context_sentences + 1, max_number_of_words_in_a_sentence)
        :param y: tuple of labels, with shape: (batch_size, )
        :return: value of loss function on batch
        """
        context_sentences = np.array(x)[:, :self.num_context_turns]              # [batch_size, 10, 50]
        response_sentences = np.array(x)[:, self.num_context_turns:].squeeze()   # [batch_size, 50]
        context_len = []   # [batch_size, 10]
        response_len = []  # [batch_size]

        # format model inputs
        # compute lens of sentences
        lens = []
        for context in context_sentences:
            context_sentences_lens = []
            for sent in context:
                context_sentences_lens.append(len(sent[sent != 0]))
            lens.append(context_sentences_lens)
        context_len += lens
        lens = []
        for response in response_sentences:
            lens.append(len(response[response != 0]))
        response_len += lens

        feed_dict = {
            self.utterance_ph: np.array(x)[:, :self.num_context_turns],
            self.all_utterance_len_ph: np.array(context_len),
            self.response_ph: np.array(x)[:, self.num_context_turns:].squeeze(),
            self.response_len_ph: np.array(response_len),
            self.y_true: np.array(y)
        }
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss
