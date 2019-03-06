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
from typing import Tuple, List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.utils import zero_pad
from logging import getLogger

from bert_dp.preprocessing import convert_examples_to_features, InputExample
from bert_dp.tokenization import FullTokenizer

logger = getLogger(__name__)


@register('bert_preprocessor')
class BertPreprocessor(Component):
    # TODO: docs

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 **kwargs):
        self.max_seq_length = max_seq_length
        vocab_file = str(expand_path(vocab_file))
        self.tokenizer = FullTokenizer(vocab_file=vocab_file,
                                       do_lower_case=do_lower_case)

    def __call__(self, texts_a, texts_b=None):
        if texts_b is None:
            texts_b = [None] * len(texts_a)
        # TODO: add unique id
        examples = [InputExample(unique_id=0, text_a=text_a, text_b=text_b)
                    for text_a, text_b in zip(texts_a, texts_b)]
        return convert_examples_to_features(examples, self.max_seq_length, self.tokenizer)


@register('bert_ner_preprocessor')
class BertNerPreprocessor(Component):
    # TODO: docs

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 max_subword_length: int = None,
                 **kwargs):
        self.max_seq_length = max_seq_length
        self.max_subword_length = max_subword_length
        vocab_file = str(expand_path(vocab_file))
        self.tokenizer = FullTokenizer(vocab_file=vocab_file,
                                       do_lower_case=do_lower_case)

    def __call__(self,
                 tokens: List[List[str]],
                 tags: List[List[str]] = None,
                 **kwargs):
        subword_tokens, subword_tok_ids, subword_masks, subword_tags = [], [], [], []
        for toks, ys in zip(tokens, tags):
            assert len(toks) == len(ys), \
                f"toks({len(toks)}) should have the same length as "\
                f" ys({len(ys)}), tokens = {toks}."
            sw_toks, sw_mask, sw_ys = self._ner_bert_tokenize(toks,
                                                              [1] * len(toks),
                                                              ys,
                                                              self.tokenizer,
                                                              self.max_subword_length)
            if self.max_seq_length is not None:
                sw_toks = sw_toks[:self.max_seq_length]
                sw_mask = sw_mask[:self.max_seq_length]
                sw_ys = sw_ys[:self.max_seq_length]

                # add [sep] if we cut it
                if sw_toks[-1] != '[SEP]':
                    sw_toks[-1] = '[SEP]'
                    sw_mask[-1] = 0
                    sw_ys[-1] = 'X'
            subword_tokens.append(sw_toks)
            subword_tok_ids.append(self.tokenizer.convert_tokens_to_ids(sw_toks))
            subword_masks.append(sw_mask)
            subword_tags.append(sw_ys)
            assert len(sw_mask) == len(sw_toks) == len(subword_tok_ids[-1]) == len(sw_ys),\
                f"length of mask({len(sw_mask)}), tokens({len(sw_toks)}),"\
                f" token ids({len(subword_tok_ids[-1])}) and ys({len(ys)})"\
                f" for tokens = `{toks}` should match"
        subword_tok_ids = zero_pad(subword_tok_ids, dtype=int, padding=0)
        subword_masks = zero_pad(subword_masks, dtype=int, padding=0)
        return subword_tokens, subword_tok_ids, subword_masks, subword_tags

    @staticmethod
    def _ner_bert_tokenize(tokens: List[str],
                           mask: List[int],
                           tags: List[str],
                           tokenizer: FullTokenizer,
                           max_subword_len: int = None) -> Tuple[List[str], List[str]]:
        tokens_subword = ['[CLS]']
        mask_subword = [0]
        tags_subword = ['X']

        for token, flag, tag in zip(tokens, mask, tags):
            subwords = tokenizer.tokenize(token)
            if not subwords or\
                    ((max_subword_len is not None) and (len(subwords) > max_subword_len)):
                tokens_subword.append('[UNK]')
                mask_subword.append(0)
                tags_subword.append('X')
            else:
                tokens_subword.extend(subwords)
                mask_subword.extend([flag] + [0] * (len(subwords) - 1))
                tags_subword.extend([tag] + ['X'] * (len(subwords) - 1))

        tokens_subword.append('[SEP]')
        mask_subword.append(0)
        tags_subword.append('X')
        return tokens_subword, mask_subword, tags_subword

