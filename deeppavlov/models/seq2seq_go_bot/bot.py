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
from typing import Dict

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.seq2seq_go_bot.network import Seq2SeqGoalOrientedBotNetwork

log = getLogger(__name__)


@register("seq2seq_go_bot")
class Seq2SeqGoalOrientedBot(NNModel):
    """
    A goal-oriented bot based on a sequence-to-sequence rnn. For implementation details see
    :class:`~deeppavlov.models.seq2seq_go_bot.network.Seq2SeqGoalOrientedBotNetwork`.
    Pretrained for :class:`~deeppavlov.dataset_readers.kvret_reader.KvretDatasetReader` dataset.

    Parameters:
        network_parameters: parameters passed to object of
            :class:`~deeppavlov.models.seq2seq_go_bot.network.Seq2SeqGoalOrientedBotNetwork` class.
        embedder: word embeddings model, see
            :doc:`deeppavlov.models.embedders </apiref/models/embedders>`.
        source_vocab: vocabulary of input tokens.
        target_vocab: vocabulary of bot response tokens.
        start_of_sequence_token: token that defines start of input sequence.
        end_of_sequence_token: token that defines end of input sequence and start of
            output sequence.
        debug: whether to display debug output.
        **kwargs: parameters passed to parent
            :class:`~deeppavlov.core.models.nn_model.NNModel` class.
    """

    def __init__(self,
                 network_parameters: Dict,
                 embedder: Component,
                 source_vocab: Component,
                 target_vocab: Component,
                 start_of_sequence_token: str,
                 end_of_sequence_token: str,
                 knowledge_base_keys,
                 save_path: str,
                 load_path: str = None,
                 debug: bool = False,
                 **kwargs) -> None:
        super().__init__(save_path=save_path, load_path=load_path, **kwargs)

        self.embedder = embedder
        self.embedding_size = embedder.dim
        self.src_vocab = source_vocab
        self.tgt_vocab = target_vocab
        self.tgt_vocab_size = len(target_vocab)
        self.kb_keys = knowledge_base_keys
        self.kb_size = len(self.kb_keys)
        self.sos_token = start_of_sequence_token
        self.eos_token = end_of_sequence_token
        self.debug = debug

        network_parameters['load_path'] = load_path
        network_parameters['save_path'] = save_path
        self.network = self._init_network(network_parameters)

    def _init_network(self, params):
        if 'target_start_of_sequence_index' not in params:
            params['target_start_of_sequence_index'] = self.tgt_vocab[self.sos_token]
        if 'target_end_of_sequence_index' not in params:
            params['target_end_of_sequence_index'] = self.tgt_vocab[self.eos_token]
        if 'source_vocab_size' not in params:
            params['source_vocab_size'] = len(self.src_vocab)
        if 'target_vocab_size' not in params:
            params['target_vocab_size'] = len(self.tgt_vocab)
        # contruct matrix of knowledge bases values embeddings
        params['knowledge_base_entry_embeddings'] = \
            [self._embed_kb_key(val) for val in self.kb_keys]
        # contrcust matrix of decoder input token embeddings (zeros for sos_token)
        dec_embs = self.embedder([[self.tgt_vocab[idx]
                                   for idx in range(self.tgt_vocab_size)]])[0]
        dec_embs[self.tgt_vocab[self.sos_token]][:] = 0.
        params['decoder_embeddings'] = dec_embs
        return Seq2SeqGoalOrientedBotNetwork(**params)

    def _embed_kb_key(self, key):
        # TODO: fasttext embedder to work with tokens
        emb = np.array(self.embedder([key.split('_')], mean=True)[0])
        if self.debug:
            log.debug("embedding key tokens='{}', embedding shape = {}"
                      .format(key.split('_'), emb.shape))
        return emb

    def train_on_batch(self, utters, history_list, kb_entry_list, responses):
        b_enc_ins, b_src_lens = [], []
        b_dec_ins, b_dec_outs, b_tgt_lens = [], [], []
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
        # b_enc_ins_np = self.src_vocab[self.sos_token] *\
        #    np.ones((batch_size, max_src_len), dtype=np.float32)
        b_enc_ins_np = np.zeros((batch_size, max_src_len, self.embedding_size),
                                dtype=np.float32)
        b_dec_ins_np = self.tgt_vocab[self.eos_token] * \
                       np.ones((batch_size, max_tgt_len), dtype=np.float32)
        b_dec_outs_np = self.tgt_vocab[self.eos_token] * \
                        np.ones((batch_size, max_tgt_len), dtype=np.float32)
        b_tgt_weights_np = np.zeros((batch_size, max_tgt_len), dtype=np.float32)
        b_kb_masks_np = np.zeros((batch_size, self.kb_size), np.float32)
        for i, (src_len, tgt_len, kb_entries) in \
                enumerate(zip(b_src_lens, b_tgt_lens, kb_entry_list)):
            b_enc_ins_np[i, :src_len] = b_enc_ins[i]
            b_dec_ins_np[i, :tgt_len] = b_dec_ins[i]
            b_dec_outs_np[i, :tgt_len] = b_dec_outs[i]
            b_tgt_weights_np[i, :tgt_len] = 1.
            if self.debug:
                if len(kb_entries) != len(set([e[0] for e in kb_entries])):
                    log.debug("Duplicates in kb_entries = {}".format(kb_entries))
            for k, v in kb_entries:
                b_kb_masks_np[i, self.kb_keys.index(k)] = 1.

        """if self.debug:
            log.debug("b_enc_ins = {}".format(b_enc_ins))
            log.debug("b_dec_ins = {}".format(b_dec_ins))
            log.debug("b_dec_outs = {}".format(b_dec_outs))
            log.debug("b_src_lens = {}".format(b_src_lens))
            log.debug("b_tgt_lens = {}".format(b_tgt_lens))
            log.debug("b_tgt_weights = {}".format(b_tgt_weights))"""

        return self.network.train_on_batch(b_enc_ins_np, b_dec_ins_np, b_dec_outs_np,
                                           b_src_lens, b_tgt_lens, b_tgt_weights_np,
                                           b_kb_masks_np)

    def _encode_context(self, tokens):
        if self.debug:
            log.debug("Context tokens = \"{}\"".format(tokens))
        # token_idxs = self.src_vocab([tokens])[0]
        # return token_idxs
        return np.array(self.embedder([tokens])[0])

    def _encode_response(self, tokens):
        if self.debug:
            log.debug("Response tokens = \"{}\"".format(tokens))
        token_idxs = []
        for token in tokens:
            if token in self.kb_keys:
                token_idxs.append(self.tgt_vocab_size + self.kb_keys.index(token))
            else:
                token_idxs.append(self.tgt_vocab[token])
        # token_idxs = self.tgt_vocab([tokens])[0]
        return ([self.tgt_vocab[self.sos_token]] + token_idxs,
                token_idxs + [self.tgt_vocab[self.eos_token]])

    def _decode_response(self, token_idxs):
        def _idx2token(idxs):
            for idx in idxs:
                if idx < self.tgt_vocab_size:
                    token = self.tgt_vocab([[idx]])[0][0]
                    if token == self.eos_token:
                        break
                    yield token
                else:
                    yield self.kb_keys[idx - self.tgt_vocab_size]

        return [list(_idx2token(utter_idxs)) for utter_idxs in token_idxs]

    def __call__(self, *batch):
        return self._infer_on_batch(*batch)

    # def _infer_on_batch(self, utters, kb_entry_list=itertools.repeat([])):
    def _infer_on_batch(self, utters, history_list, kb_entry_list):
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
        b_enc_ins_np = np.zeros((batch_size, max_src_len, self.embedding_size),
                                dtype=np.float32)
        b_kb_masks_np = np.zeros((batch_size, self.kb_size), dtype=np.float32)
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
