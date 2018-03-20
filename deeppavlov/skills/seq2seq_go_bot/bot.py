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
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.encoders.bow import BoWEncoder
from deeppavlov.models.tokenizers.spacy_tokenizer import StreamSpacyTokenizer
from deeppavlov.skills.seq2seq_go_bot.network import Seq2SeqGoalOrientedBotNetwork
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("seq2seq_go_bot")
class Seq2SeqGoalOrientedBot(NNModel):
    def __init__(self,
                 end_of_sequence_token,
                 start_of_sequence_token,
                 tokenizer: Type = StreamSpacyTokenizer,
                 network: Type = Seq2SeqGoalOrientedBotNetwork,
                 source_vocab: Type = DefaultVocabulary,
                 target_vocab: Type = DefaultVocabulary,
                 bow_encoder: Type = BoWEncoder,
                 debug=False,
                 save_path=None,
                 **kwargs):

        super().__init__(save_path=save_path, mode=kwargs['mode'])

        self.sos_token = start_of_sequence_token
        self.eos_token = end_of_sequence_token
        self.tokenizer = tokenizer
        self.network = network
        self.src_vocab = source_vocab
        self.tgt_vocab = target_vocab
        self.bow_encoder = bow_encoder
        #self.embedder = embedder
        self.debug = debug

    def train_on_batch(self, x, y):
        b_enc_ins, b_src_seq_lens = [], []
        b_dec_ins, b_dec_outs, b_tgt_seq_lens, b_tgt_weights = [], [], [], []
        for d_contexts, d_responses in zip(x, y):
            for context, response in zip(d_contexts, d_responses):

                enc_in = self._encode_context(context['text'])
                b_enc_ins.append(enc_in)
                b_src_seq_lens.append(len(enc_in))

                dec_in, dec_out = self._encode_response(response['text'])
                b_dec_ins.append(dec_in)
                b_dec_outs.append(dec_out)
                b_tgt_seq_lens.append(len(dec_out))
                b_tgt_weights.append([1] * len(dec_out))

        # Sequence padding
        max_src_seq_len = max(b_src_seq_lens)
        max_tgt_seq_len = max(b_tgt_seq_lens)
        for i, (src_seq_len, tgt_seq_len) in enumerate(zip(b_src_seq_lens, b_tgt_seq_lens)):
            src_padd_len = max_src_seq_len - src_seq_len
            tgt_padd_len = max_tgt_seq_len - tgt_seq_len
            b_enc_ins[i].extend([self.src_vocab[self.eos_token]] * src_padd_len)
            b_dec_ins[i].extend([self.tgt_vocab[self.eos_token]] * tgt_padd_len)
            b_dec_outs[i].extend([self.tgt_vocab[self.eos_token]] * tgt_padd_len)
            b_tgt_weights[i].extend([0] * tgt_padd_len)

        self.network.train_on_batch(b_enc_ins, b_dec_ins, b_dec_outs,
                                    b_src_seq_lens, b_tgt_seq_lens, b_tgt_weights)

    def _encode_context(self, context):
        tokens = self.tokenizer([context])[0]
        if self.debug:
            log.debug("Context tokens = \"{}\"".format(tokens))

        token_idxs = self.src_vocab(tokens)
        return token_idxs

    def _encode_response(self, response):
        tokens = self.tokenizer([response])[0]
        if self.debug:
            log.debug("Response tokens = \"{}\"".format(tokens))

        token_idxs = self.tgt_vocab(tokens)
        return ([self.tgt_vocab[self.sos_token]] + token_idxs,
                token_idxs + [self.tgt_vocab[self.eos_token]])

    def __call__(self, batch):
        if isinstance(batch[0], str):
            return self._infer_on_batch([{'text': x} for x in batch])
        return self._infer_on_batch(batch)

    def _infer_on_batch(self, contexts):
# TODO: history as input
        b_enc_ins, b_src_seq_lens = [], []
        d_idxs = []
        for d_contexts in contexts:
            d_idxs.append({'start': len(d_idxs), 'end': len(d_idxs) + len(d_contexts)})
            for context in d_contexts:
                enc_in = self._encode_context(context['text'])
                b_enc_ins.append(enc_in)
                b_src_seq_lens.append(len(enc_in))

        # Sequence padding
        max_src_seq_len = max(b_src_seq_lens)
        for i, src_seq_len in enumerate(b_src_seq_lens):
            src_padd_len = max_src_seq_len - src_seq_len
            b_enc_ins[i].extend([self.src_vocab[self.eos_token]] * src_padd_len)

        pred_idxs = self.network(b_enc_ins, b_src_seq_lens)
        if self.debug:
            print(pred_idxs, len(pred_idxs), len(pred_idxs[0]))
        preds = []
        for d_idx in d_idxs:
            d_pred_idxs = pred_idxs[d_idx['start']:d_idx['end']]
            preds.append([])
            for utter_idxs in d_pred_idxs:
                preds[-1].append(' '.join(self.tgt_vocab(list(utter_idxs))))
            if self.debug:
                print("Dialog predictions = \"{}\"".format(preds[-1]))
        return preds

    def save(self):
        self.network.save()

    def load(self):
        pass
