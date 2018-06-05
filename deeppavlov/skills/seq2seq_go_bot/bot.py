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
from typing import Type

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.data.vocab import DefaultVocabulary
from deeppavlov.skills.seq2seq_go_bot.network import Seq2SeqGoalOrientedBotNetwork
# from deeppavlov.skills.seq2seq_go_bot.wrapper import PerItemWrapper
from deeppavlov.skills.seq2seq_go_bot.dialog_state import DialogState
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("seq2seq_go_bot")
class Seq2SeqGoalOrientedBot(NNModel):
    def __init__(self,
                 end_of_sequence_token,
                 start_of_sequence_token,
                 network: Type = Seq2SeqGoalOrientedBotNetwork,
                 source_vocab: Type = DefaultVocabulary,
                 target_vocab: Type = DefaultVocabulary,
                 knowledge_base_keys: Type = list,
                 debug=False,
                 save_path=None,
                 **kwargs):

        super().__init__(save_path=save_path, mode=kwargs['mode'])

        self.sos_token = start_of_sequence_token
        self.eos_token = end_of_sequence_token
        self.network = network
        self.src_vocab = source_vocab
        self.tgt_vocab = target_vocab
        self.tgt_vocab_size = len(target_vocab)
        self.kb_keys = knowledge_base_keys
        self.kb_size = len(self.kb_keys)
        #self.embedder = embedder
        self.debug = debug

    def train_on_batch(self, utters, history_list, kb_entry_list, responses):
        b_enc_ins, b_src_lens = [], []
        b_dec_ins, b_dec_outs, b_tgt_lens, b_tgt_weights = [], [], [], []
        for x_tokens, history, y_tokens in zip(utters, history_list, responses):
            x_tokens = history + x_tokens
            enc_in = self._encode_context(x_tokens)
            b_enc_ins.append(enc_in)
            b_src_lens.append(len(enc_in))

            dec_in, dec_out = self._encode_response(y_tokens)
            b_dec_ins.append(dec_in)
            b_dec_outs.append(dec_out)
            b_tgt_lens.append(len(dec_out))

        # Sequence padding
        batch_size = len(b_enc_ins)
        max_src_len = max(b_src_lens)
        max_tgt_len = max(b_tgt_lens)
        b_enc_ins_np = np.ones((batch_size, max_src_len)) * self.src_vocab[self.sos_token]
        b_dec_ins_np = np.ones((batch_size, max_tgt_len)) * self.tgt_vocab[self.eos_token]
        b_dec_outs_np = np.ones((batch_size, max_tgt_len)) * self.tgt_vocab[self.eos_token]
        b_tgt_weights_np = np.zeros((batch_size, max_tgt_len))
        b_kb_masks_np = np.zeros((batch_size, self.kb_size), np.float32)
        for i, (src_len, tgt_len, kb_entries) in \
                enumerate(zip(b_src_lens, b_tgt_lens, kb_entry_list)):
            b_enc_ins_np[i, :src_len] = b_enc_ins[i]
            b_dec_ins_np[i, :tgt_len] = b_dec_ins[i]
            b_dec_outs_np[i, :tgt_len] = b_dec_outs[i]
            b_tgt_weights_np[i, :tgt_len] = 1
            if self.debug:
                if len(kb_entries) != len(set([e[0] for e in kb_entries])):
                    print("Duplicates in kb_entries = {}".format(kb_entries))
            for k, v in kb_entries:
                b_kb_masks_np[i, self.kb_keys.index(k)] = 1.

        """if self.debug:
            log.debug("b_enc_ins = {}".format(b_enc_ins))
            log.debug("b_dec_ins = {}".format(b_dec_ins))
            log.debug("b_dec_outs = {}".format(b_dec_outs))
            log.debug("b_src_lens = {}".format(b_src_lens))
            log.debug("b_tgt_lens = {}".format(b_tgt_lens))
            log.debug("b_tgt_weights = {}".format(b_tgt_weights))"""

        self.network.train_on_batch(b_enc_ins_np, b_dec_ins_np, b_dec_outs_np,
                                    b_src_lens, b_tgt_lens, b_tgt_weights_np,
                                    b_kb_masks_np)

    def _encode_context(self, tokens):
        if self.debug:
            log.debug("Context tokens = \"{}\"".format(tokens))
        token_idxs = self.src_vocab(tokens)
        return token_idxs

    def _encode_response(self, tokens):
        if self.debug:
            log.debug("Response tokens = \"{}\"".format(y_tokens))
        token_idxs = []
        for token in tokens:
            if token in self.tgt_vocab:
                token_idxs.append(self.tgt_vocab[token])
            else:
                if token not in self.kb_keys:
                    print("token = {}, tokens = {}".format(token, tokens))
                token_idxs.append(self.tgt_vocab_size + self.kb_keys.index(token))
        # token_idxs = self.tgt_vocab(tokens)
        return ([self.tgt_vocab[self.sos_token]] + token_idxs,
                token_idxs + [self.tgt_vocab[self.eos_token]])

    def _decode_response(self, token_idxs):
        def _idx2token(idxs):
            for idx in idxs:
                if idx < self.tgt_vocab_size:
                    token = self.tgt_vocab([idx])[0]
                    if token == self.eos_token:
                        break
                    yield token
                else:
                    yield self.kb_keys[idx - self.tgt_vocab_size]
        return [list(_idx2token(utter_idxs)) for utter_idxs in token_idxs]

    def __call__(self, *batch):
        return self._infer_on_batch(*batch)

    #def _infer_on_batch(self, utters, kb_entry_list=itertools.repeat([])):
    def _infer_on_batch(self, utters, history_list, kb_entry_list):
# TODO: history as input
        b_enc_ins, b_src_lens = [], []
        if (len(utters) == 1) and not utters[0]:
            utters = [['hi']]
        for utter, history in zip(utters, history_list):
            utter = history + utter
            enc_in = self._encode_context(utter)

            b_enc_ins.append(enc_in)
            b_src_lens.append(len(enc_in))

        # Sequence padding
        batch_size = len(b_enc_ins)
        max_src_len = max(b_src_lens)
        b_enc_ins_np = np.ones((batch_size, max_src_len)) * self.src_vocab[self.sos_token]
        b_kb_masks_np = np.zeros((batch_size, self.kb_size), np.float32)
        for i, (src_len, kb_entries) in enumerate(zip(b_src_lens, kb_entry_list)):
            b_enc_ins_np[i, :src_len] = b_enc_ins[i]
            if self.debug:
                log.debug("infer: kb_entries = {}".format(kb_entries))
            for k, v in kb_entries:
                b_kb_masks_np[i, self.kb_keys.index(k)] = 1.

        pred_idxs = self.network(b_enc_ins_np, b_src_lens, b_kb_masks_np)
        preds = self._decode_response(pred_idxs)
        if self.debug:
            log.debug("Dialog prediction = \"{}\"".format(preds[-1]))
        return preds

    def save(self):
        self.network.save()

    def load(self):
        pass
