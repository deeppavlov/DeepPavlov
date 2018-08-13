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

import itertools

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("seq2seq_go_bot")
class Seq2SeqGoalOrientedBot(NNModel):
    """
    A goal-oriented bot based on a sequence-to-sequence rnn. For implementation details see :class:`~deeppavlov.models.seq2seq_go_bot.network.Seq2SeqGoalOrientedBotNetwork`. Pretrained for :class:`~deeppavlov.dataset_readers.kvret_reader.KvretDatasetReader` dataset.

    Parameters:
        network: object of :class:`~deeppavlov.models.seq2seq_go_bot.network.Seq2SeqGoalOrientedBotNetwork` class.
        source_vocab: vocabulary of input tokens.
        target_vocab: vocabulary of bot response tokens.
        start_of_sequence_token: token that defines start of input sequence.
        end_of_sequence_token: token that defines end of input sequence and start of output sequence.
        debug: whether to display debug output.
        **kwargs: parameters passed to parent :class:`~deeppavlov.core.models.nn_model.NNModel` class.
    """
    def __init__(self,
                 network: Component,
                 source_vocab: Component,
                 target_vocab: Component,
                 start_of_sequence_token: str,
                 end_of_sequence_token: str,
                 debug: bool = False,
                 save_path: str = None,
                 **kwargs) -> None:
        super().__init__(save_path=save_path, **kwargs)

        self.network = network
        self.src_vocab = source_vocab
        self.tgt_vocab = target_vocab
        self.sos_token = start_of_sequence_token
        self.eos_token = end_of_sequence_token
        self.debug = debug

    def train_on_batch(self, *batch):
        b_enc_ins, b_src_lens = [], []
        b_dec_ins, b_dec_outs, b_tgt_lens, b_tgt_weights = [], [], [], []
        for x_tokens, dialog_id, y_tokens in zip(*batch):

            enc_in = self._encode_context(x_tokens, dialog_id)
            b_enc_ins.append(enc_in)
            b_src_lens.append(len(enc_in))

            dec_in, dec_out = self._encode_response(y_tokens)
            b_dec_ins.append(dec_in)
            b_dec_outs.append(dec_out)
            b_tgt_lens.append(len(dec_out))
            b_tgt_weights.append([1] * len(dec_out))

        # Sequence padding
        max_src_len = max(b_src_lens)
        max_tgt_len = max(b_tgt_lens)
        for i, (src_len, tgt_len) in enumerate(zip(b_src_lens, b_tgt_lens)):
            src_padd_len = max_src_len - src_len
            tgt_padd_len = max_tgt_len - tgt_len
            b_enc_ins[i].extend([self.src_vocab[self.sos_token]] * src_padd_len)
            b_dec_ins[i].extend([self.tgt_vocab[self.eos_token]] * tgt_padd_len)
            b_dec_outs[i].extend([self.tgt_vocab[self.eos_token]] * tgt_padd_len)
            b_tgt_weights[i].extend([0] * tgt_padd_len)

        self.network.train_on_batch(b_enc_ins, b_dec_ins, b_dec_outs,
                                    b_src_lens, b_tgt_lens, b_tgt_weights)

    def _encode_context(self, tokens, dialog_id=None):
        if self.debug:
            log.debug("Context tokens = \"{}\"".format(tokens))
        token_idxs = self.src_vocab(tokens)
        return token_idxs

    def _encode_response(self, tokens):
        if self.debug:
            log.debug("Response tokens = \"{}\"".format(tokens))
        token_idxs = self.tgt_vocab(tokens)
        return ([self.tgt_vocab[self.sos_token]] + token_idxs,
                token_idxs + [self.tgt_vocab[self.eos_token]])

    def __call__(self, *batch):
        return self._infer_on_batch(*batch)

    def _infer_on_batch(self, utters, dialog_ids=itertools.repeat(None)):
        def _filter(tokens):
            for t in tokens:
                if t == self.eos_token:
                    break
                yield t
# TODO: history as input
        b_enc_ins, b_src_lens = [], []
        if (len(utters) == 1) and not utters[0]:
            utters = [['hi']]
        for utter, dialog_id in zip(utters, dialog_ids):
            enc_in = self._encode_context(utter, dialog_id)
            b_enc_ins.append(enc_in)
            b_src_lens.append(len(enc_in))

        # Sequence padding
        max_src_len = max(b_src_lens)
        for i, src_len in enumerate(b_src_lens):
            src_padd_len = max_src_len - src_len
            b_enc_ins[i].extend([self.src_vocab[self.eos_token]] * src_padd_len)

        pred_idxs = self.network(b_enc_ins, b_src_lens)
        preds = [list(_filter(self.tgt_vocab(utter_idxs)))
                 for utter_idxs in pred_idxs]
        if self.debug:
            log.debug("Dialog prediction = \"{}\"".format(preds[-1]))
        return preds

    def save(self):
        self.network.save()

    def load(self):
        pass
