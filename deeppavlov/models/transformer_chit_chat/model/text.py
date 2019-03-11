#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pytorch_pretrained_bert import BertTokenizer


class BertBPEVocab:
    we = '##'

    pad_token = '[PAD]'
    bos_token = '[S]'
    eos_token = '[/S]'
    info_bos = '[I]'
    info_eos = '[/I]'
    talker1_bos = '[T1]'
    talker1_eos = '[/T1]'
    talker2_bos = '[T2]'
    talker2_eos = '[/T2]'
    talker2_eos = '[/T2]'
    unk_token = '[UNK]'
    cls_token = '[CLS]'
    sep_token = '[SEP]'
    mask_token = '[MASK]'

    @staticmethod
    def from_files(vocab_path, *args, **kwargs):
        tokenizer = BertTokenizer(vocab_path, do_lower_case=False)
        return BertBPEVocab(tokenizer, *args, **kwargs)

    @staticmethod
    def get_pairs(string):
        if len(string) < 2:
            return set()

        return set(zip(string[:-1], string[1:]))

    def __init__(self, tokenizer):
        # TODO: add check for special tokens
        tokenizer
        self.spec_tokens = [BertBPEVocab.pad_token, BertBPEVocab.bos_token, BertBPEVocab.eos_token,
                            BertBPEVocab.info_bos, BertBPEVocab.info_eos, BertBPEVocab.talker1_bos,
                            BertBPEVocab.talker1_eos, BertBPEVocab.talker2_bos, BertBPEVocab.talker2_eos,
                            BertBPEVocab.unk_token, BertBPEVocab.cls_token, BertBPEVocab.sep_token,
                            BertBPEVocab.mask_token,
                            ]
        self.token2id = tokenizer.vocab
        self.id2token = tokenizer.ids_to_tokens
        self.tokenizer = tokenizer
        self.cache = {}

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return len(self.spec_tokens)

    @property
    def special_tokens_ids(self):
        return [self.token2id[t] for t in self.spec_tokens]

    @property
    def pad_id(self):
        return self.token2id[BertBPEVocab.pad_token]

    @property
    def bos_id(self):
        return self.token2id[BertBPEVocab.bos_token]

    @property
    def eos_id(self):
        return self.token2id[BertBPEVocab.eos_token]

    @property
    def info_bos_id(self):
        return self.token2id[BertBPEVocab.info_bos]

    @property
    def info_eos_id(self):
        return self.token2id[BertBPEVocab.info_eos]

    @property
    def talker1_bos_id(self):
        return self.token2id[BertBPEVocab.talker1_bos]

    @property
    def talker1_eos_id(self):
        return self.token2id[BertBPEVocab.talker1_eos]

    @property
    def talker2_bos_id(self):
        return self.token2id[BertBPEVocab.talker2_bos]

    @property
    def talker2_eos_id(self):
        return self.token2id[BertBPEVocab.talker2_eos]

    @property
    def sep_id(self):
        return self.token2id[BertBPEVocab.sep_token]

    def string2ids(self, string):
        tokens = self.tokenizer.tokenize(string)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # tokens = self.tokenizer.convert_ids_to_tokens(tokens)
        return ids

    def ids2string(self, ids):
        bpe_tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return ' '.join(bpe_tokens).replace(' '+BertBPEVocab.we, '')
