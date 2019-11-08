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

from logging import getLogger
from typing import List, Dict, Tuple

import numpy as np

from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.models.ranking.siamese_model import SiameseModel

log = getLogger(__name__)


class TensorflowBaseMatchingModel(TFModel, SiameseModel):
    """
    Base class for ranking models that uses context-response matching schemes.

    Note:
        Tensorflow session variable already presents as self.sess attribute
        (derived from TFModel and initialized by Chainer)

    Args:
        batch_size (int): a number of samples in a batch.
        num_context_turns (int): a number of ``context`` turns in data samples.
        mean_oov (bool): whether to set mean embedding of all tokens. By default: True.
        use_logits (bool): whether to use raw logits as outputs instead of softmax predictions

    """

    def __init__(self,
                 batch_size: int,
                 num_context_turns: int = 10,
                 mean_oov: bool = True,
                 use_logits: bool = False,
                 *args,
                 **kwargs):
        super(TensorflowBaseMatchingModel, self).__init__(batch_size=batch_size, num_context_turns=num_context_turns,
                                                          *args, **kwargs)
        self.use_logits = use_logits
        if mean_oov:
            self.emb_matrix[1] = np.mean(self.emb_matrix[2:],
                                         axis=0)  # set mean embedding for OOV token at the 2nd index

    def _append_sample_to_batch_buffer(self, sample: List[np.ndarray], buf: List[Tuple]) -> int:
        """

        Args:
            sample (List[nd.array]): samples generator
            buf (List[Tuple]) : List of samples with model inputs each:
                [( context, context_len, response, response_len ), ( ... ), ... ].
        Returns:
             a number of candidate responses
        """
        #
        batch_buffer_context = []  # [batch_size, 10, 50]
        batch_buffer_context_len = []  # [batch_size, 10]
        batch_buffer_response = []  # [batch_size, 50]
        batch_buffer_response_len = []  # [batch_size]

        context_sentences = sample[:self.num_context_turns]
        response_sentences = sample[self.num_context_turns:]

        # Format model inputs:
        # 4 model inputs

        # 1. Token indices for context
        batch_buffer_context += [context_sentences] * len(response_sentences)
        # 2. Token indices for response
        batch_buffer_response += list(response_sentences)
        # 3. Lens of context sentences
        lens = []
        for context in [context_sentences] * len(response_sentences):
            context_sentences_lens = []
            for sent in context:
                context_sentences_lens.append(len(sent[sent != 0]))
            lens.append(context_sentences_lens)
        batch_buffer_context_len += lens
        # 4. Lens of response sentences
        lens = []
        for response_sent in response_sentences:
            lens.append(len(response_sent[response_sent != 0]))
        batch_buffer_response_len += lens

        for i in range(len(batch_buffer_context)):
            buf.append(tuple((
                batch_buffer_context[i],
                batch_buffer_context_len[i],
                batch_buffer_response[i],
                batch_buffer_response_len[i]
            )))

        return len(response_sentences)

    def _make_batch(self, batch: List[Tuple[List[np.ndarray], List, np.ndarray, int]]) -> Dict:
        """
        The function for formatting model inputs

        Args:
            batch (List[Tuple[np.ndarray]]): List of samples with model inputs each:
                [( context, context_len, response, response_len ), ( ... ), ... ].
        Returns:
            Dict: feed_dict to feed a model
        """
        input_context = []
        input_context_len = []
        input_response = []
        input_response_len = []

        # format model inputs as numpy arrays
        for sample in batch:
            input_context.append(sample[0])
            input_context_len.append(sample[1])
            input_response.append(sample[2])
            input_response_len.append(sample[3])

        return {
            self.utterance_ph: np.array(input_context),
            self.all_utterance_len_ph: np.array(input_context_len),
            self.response_ph: np.array(input_response),
            self.response_len_ph: np.array(input_response_len)
        }

    def _predict_on_batch(self, batch: Dict) -> np.ndarray:
        """
        Run a model with the batch of inputs.
        The function returns a list of predictions for the batch in numpy format

        Args:
            batch (Dict): feed_dict that contains a batch with inputs for a model

        Returns:
            nd.array: predictions for the batch (raw logits or softmax outputs)
        """
        if self.use_logits:
            return self.sess.run(self.logits, feed_dict=batch)[:, 1]
        else:
            return self.sess.run(self.y_pred, feed_dict=batch)[:, 1]

    def _train_on_batch(self, batch: Dict, y: List[int]) -> float:
        """
        The function is for formatting of feed_dict used as an input for a model
        Args:
            batch (Dict): feed_dict that contains a batch with inputs for a model (except ground truth labels)
            y (List(int)): list of ground truth labels

        Returns:
            float: value of mean loss on the batch
        """
        batch.update({self.y_true: np.array(y)})
        return self.sess.run([self.loss, self.train_op], feed_dict=batch)[0]  # return the first item aka loss
