# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
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

import numpy as np
from typing import List, Iterable

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.component import Component
from deeppavlov.models.ranking.deep_attention_matching_network_use_transformer import DAMNetworkUSETransformer
from deeppavlov.core.common.registry import register

log = get_logger(__name__)


@register('matching_predictor_use')
class MatchingPredictorUSE(Component):
    """The class for ranking of the response given N context turns
    using the trained DAM-USE-T neural network in the ``interact`` mode.

    Args:
        model (:class:`~deeppavlov.models.ranking.deep_attention_matching_network_use_transformer.DAMNetworkUSETransformer`):
        DAM-USE-T model instance.
        num_context_turns (int): A number N of ``context`` turns in data samples.
        max_sequence_length (int): A maximum length of text sequences in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        *args, **kwargs: Other parameters.
    """

    def __init__(self,
                 model: DAMNetworkUSETransformer,
                 num_context_turns: int = 10,
                 max_sequence_length: int = 50,
                 *args, **kwargs) -> None:

        super(MatchingPredictorUSE, self).__init__()

        self.num_context_turns = num_context_turns
        self.max_sequence_length = max_sequence_length
        self.model = model

    def __call__(self, batch: Iterable[List[np.ndarray]]) -> List[str]:
        """
        Overrides __call__ method.

        Args:
            batch (Iterable): A batch of one sample, preprocessed and padded to ``num_context_turns`` sentences

        Return:
             list of verdict messages
        """
        sample = next(batch)
        try:
            next(batch)
            log.error("It is not intended to use the `%s` with the batch size greater then 1." % self.__class__)
        except StopIteration:
            pass

        preproc_sample = []
        for i in range(len(sample)):
            if i < len(sample) // 2:
                preproc_sample.append(np.asarray(sample[i]))
            else:
                preproc_sample.append(sample[i])

        y_pred = []
        buf = []
        n_responses = len(preproc_sample[self.num_context_turns:len(preproc_sample) // 2])
        if len(preproc_sample[:self.num_context_turns]) != self.num_context_turns:
            log.error("Number of context sentences should be equal to %s" % self.num_context_turns)
            return ["Number of context sentences should be equal to %s" % self.num_context_turns]

        self.model._append_sample_to_batch_buffer(preproc_sample, buf)
        if len(buf) >= self.model.batch_size:
            for i in range(len(buf) // self.model.batch_size):
                # 1. USE Graph
                fd = self.model._make_batch(buf[i * self.model.batch_size:(i + 1) * self.model.batch_size], graph="use")
                context_emb, response_emb = self.model._predict_on_batch(fd, graph="use")

                # 2. MAIN Graph
                fd = self.model._make_batch(buf[i * self.model.batch_size:(i + 1) * self.model.batch_size], graph="main")
                fd.update({
                    self.model.context_sent_emb_ph: context_emb,
                    self.model.response_sent_emb_ph: response_emb
                })
                yp = self.model._predict_on_batch(fd, graph="main")
                y_pred += list(yp)
            lenb = len(buf) % self.model.batch_size
            if lenb != 0:
                buf = buf[-lenb:]
            else:
                buf = []
        if len(buf) != 0:
            # 1. USE Graph
            fd = self.model._make_batch(buf, graph="use")
            context_emb, response_emb = self.model._predict_on_batch(fd, graph="use")

            # 2. MAIN Graph
            fd = self.model._make_batch(buf, graph="main")
            fd.update({
                self.model.context_sent_emb_ph: context_emb,
                self.model.response_sent_emb_ph: response_emb
            })
            yp = self.model._predict_on_batch(fd, graph="main")
            y_pred += list(yp)
        y_pred = np.asarray(y_pred)
        # reshape to [batch_size, n_responses] if needed (n_responses > 1)
        y_pred = np.reshape(y_pred, (1, n_responses)) if n_responses > 1 else y_pred
        return ["{:.5f}".format(v) for v in y_pred[0]]

    def reset(self) -> None:
        pass

    def process_event(self) -> None:
        pass
