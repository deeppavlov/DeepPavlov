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

import re
import random
from logging import getLogger
from pathlib import Path
import torch
from typing import Tuple, List, Optional, Union, Dict, Any

from transformers import AutoTokenizer, BertTokenizer
from transformers.data.processors.utils import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component
from deeppavlov.models.preprocessors.mask import Mask

log = getLogger(__name__)


@register('torch_transformers_preprocessor')
class TorchTransformersPreprocessor(Component):
    """Tokenize text on subtokens, encode subtokens with their indices, create tokens and segment masks.

    Check details in :func:`bert_dp.preprocessing.convert_examples_to_features` function.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        return_tokens: whether to return tuple of inputfeatures and tokens, or only inputfeatures

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        return_tokens: whether to return tuple of inputfeatures and tokens, or only inputfeatures
        tokenizer: instance of Bert FullTokenizer

    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, texts_a: List[str], texts_b: Optional[List[str]] = None) -> Union[
        List[InputFeatures], Tuple[List[InputFeatures], List[List[str]]]]:
        """Tokenize and create masks.

        texts_a and texts_b are separated by [SEP] token

        Args:
            texts_a: list of texts,
            texts_b: list of texts, it could be None, e.g. single sentence classification task

        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
        """

        if texts_b is None:
            texts_b = [None] * len(texts_a)

        input_features = []
        tokens = []
        for text_a, text_b in zip(texts_a, texts_b):
            encoded_dict = self.tokenizer.encode_plus(
                text=text_a, text_pair=text_b, add_special_tokens=True, max_length=self.max_seq_length,
                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')

            if 'token_type_ids' not in encoded_dict:
                encoded_dict['token_type_ids'] = torch.tensor([0])

            curr_features = InputFeatures(input_ids=encoded_dict['input_ids'],
                                          attention_mask=encoded_dict['attention_mask'],
                                          token_type_ids=encoded_dict['token_type_ids'],
                                          label=None)
            input_features.append(curr_features)
            if self.return_tokens:
                tokens.append(self.tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0]))

        if self.return_tokens:
            return input_features, tokens
        else:
            return input_features


@register('torch_transformers_ner_preprocessor')
class TorchTransformersNerPreprocessor(Component):
    """
    Takes tokens and splits them into bert subtokens, encodes subtokens with their indices.
    Creates a mask of subtokens (one for the first subtoken, zero for the others).

    If tags are provided, calculates tags for subtokens.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: replace token to <unk> if it's length is larger than this
            (defaults to None, which is equal to +infinity)
        token_masking_prob: probability of masking token while training
        provide_subword_tags: output tags for subwords or for words
        subword_mask_mode: subword to select inside word tokens, can be "first" or "last"
            (default="first")

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: rmax lenght of a bert subtoken
        tokenizer: instance of Bert FullTokenizer
    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = False,
                 max_seq_length: int = 512,
                 max_subword_length: int = None,
                 token_masking_prob: float = 0.0,
                 provide_subword_tags: bool = False,
                 subword_mask_mode: str = "first",
                 **kwargs):
        self._re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.provide_subword_tags = provide_subword_tags
        self.mode = kwargs.get('mode')
        self.max_seq_length = max_seq_length
        self.max_subword_length = max_subword_length
        self.subword_mask_mode = subword_mask_mode
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=True)
        self.token_masking_prob = token_masking_prob

    def __call__(self,
                 tokens: Union[List[List[str]], List[str]],
                 tags: List[List[str]] = None,
                 **kwargs):
        tokens_offsets_batch = [[] for _ in tokens]
        if isinstance(tokens[0], str):
            tokens_batch = []
            tokens_offsets_batch = []
            for s in tokens:
                tokens_list = []
                tokens_offsets_list = []
                for elem in re.finditer(self._re_tokenizer, s):
                    tokens_list.append(elem[0])
                    tokens_offsets_list.append((elem.start(), elem.end()))
                tokens_batch.append(tokens_list)
                tokens_offsets_batch.append(tokens_offsets_list)
            tokens = tokens_batch
        subword_tokens, subword_tok_ids, startofword_markers, subword_tags = [], [], [], []
        for i in range(len(tokens)):
            toks = tokens[i]
            ys = ['O'] * len(toks) if tags is None else tags[i]
            assert len(toks) == len(ys), \
                f"toks({len(toks)}) should have the same length as ys({len(ys)})"
            sw_toks, sw_marker, sw_ys = \
                self._ner_bert_tokenize(toks,
                                        ys,
                                        self.tokenizer,
                                        self.max_subword_length,
                                        mode=self.mode,
                                        subword_mask_mode=self.subword_mask_mode,
                                        token_masking_prob=self.token_masking_prob)
            if self.max_seq_length is not None:
                if len(sw_toks) > self.max_seq_length:
                    raise RuntimeError(f"input sequence after bert tokenization"
                                       f" shouldn't exceed {self.max_seq_length} tokens.")
            subword_tokens.append(sw_toks)
            subword_tok_ids.append(self.tokenizer.convert_tokens_to_ids(sw_toks))
            startofword_markers.append(sw_marker)
            subword_tags.append(sw_ys)
            assert len(sw_marker) == len(sw_toks) == len(subword_tok_ids[-1]) == len(sw_ys), \
                f"length of sow_marker({len(sw_marker)}), tokens({len(sw_toks)})," \
                f" token ids({len(subword_tok_ids[-1])}) and ys({len(ys)})" \
                f" for tokens = `{toks}` should match"

        subword_tok_ids = zero_pad(subword_tok_ids, dtype=int, padding=0)
        startofword_markers = zero_pad(startofword_markers, dtype=int, padding=0)
        attention_mask = Mask()(subword_tokens)

        if tags is not None:
            if self.provide_subword_tags:
                return tokens, subword_tokens, subword_tok_ids, \
                       attention_mask, startofword_markers, subword_tags
            else:
                nonmasked_tags = [[t for t in ts if t != 'X'] for ts in tags]
                for swts, swids, swms, ts in zip(subword_tokens,
                                                 subword_tok_ids,
                                                 startofword_markers,
                                                 nonmasked_tags):
                    if (len(swids) != len(swms)) or (len(ts) != sum(swms)):
                        log.warning('Not matching lengths of the tokenization!')
                        log.warning(f'Tokens len: {len(swts)}\n Tokens: {swts}')
                        log.warning(f'Markers len: {len(swms)}, sum: {sum(swms)}')
                        log.warning(f'Masks: {swms}')
                        log.warning(f'Tags len: {len(ts)}\n Tags: {ts}')
                return tokens, subword_tokens, subword_tok_ids, \
                       attention_mask, startofword_markers, nonmasked_tags
        return tokens, subword_tokens, subword_tok_ids, startofword_markers, attention_mask, tokens_offsets_batch

    @staticmethod
    def _ner_bert_tokenize(tokens: List[str],
                           tags: List[str],
                           tokenizer: AutoTokenizer,
                           max_subword_len: int = None,
                           mode: str = None,
                           subword_mask_mode: str = "first",
                           token_masking_prob: float = None) -> Tuple[List[str], List[int], List[str]]:
        do_masking = (mode == 'train') and (token_masking_prob is not None)
        do_cutting = (max_subword_len is not None)
        tokens_subword = ['[CLS]']
        startofword_markers = [0]
        tags_subword = ['X']
        for token, tag in zip(tokens, tags):
            token_marker = int(tag != 'X')
            subwords = tokenizer.tokenize(token)
            if not subwords or (do_cutting and (len(subwords) > max_subword_len)):
                tokens_subword.append('[UNK]')
                startofword_markers.append(token_marker)
                tags_subword.append(tag)
            else:
                if do_masking and (random.random() < token_masking_prob):
                    tokens_subword.extend(['[MASK]'] * len(subwords))
                else:
                    tokens_subword.extend(subwords)
                if subword_mask_mode == "last":
                    startofword_markers.extend([0] * (len(subwords) - 1) + [token_marker])
                else:
                    startofword_markers.extend([token_marker] + [0] * (len(subwords) - 1))
                tags_subword.extend([tag] + ['X'] * (len(subwords) - 1))

        tokens_subword.append('[SEP]')
        startofword_markers.append(0)
        tags_subword.append('X')
        return tokens_subword, startofword_markers, tags_subword


@register('torch_bert_ranker_preprocessor')
class TorchBertRankerPreprocessor(TorchTransformersPreprocessor):
    """Tokenize text to sub-tokens, encode sub-tokens with their indices, create tokens and segment masks for ranking.

    Builds features for a pair of context with each of the response candidates.
    """

    def __call__(self, batch: List[List[str]]) -> List[List[InputFeatures]]:
        """Tokenize and create masks.

        Args:
            batch: list of elements where the first element represents the batch with contexts
                and the rest of elements represent response candidates batches

        Returns:
            list of feature batches with subtokens, subtoken ids, subtoken mask, segment mask.
        """

        if isinstance(batch[0], str):
            batch = [batch]

        cont_resp_pairs = []
        if len(batch[0]) == 1:
            contexts = batch[0]
            responses_empt = [None] * len(batch)
            cont_resp_pairs.append(zip(contexts, responses_empt))
        else:
            contexts = [el[0] for el in batch]
            for i in range(1, len(batch[0])):
                responses = []
                for el in batch:
                    responses.append(el[i])
                cont_resp_pairs.append(zip(contexts, responses))

        input_features = []

        for s in cont_resp_pairs:
            sub_list_features = []
            for context, response in s:
                encoded_dict = self.tokenizer.encode_plus(
                    text=context, text_pair=response, add_special_tokens=True, max_length=self.max_seq_length,
                    pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')

                curr_features = InputFeatures(input_ids=encoded_dict['input_ids'],
                                              attention_mask=encoded_dict['attention_mask'],
                                              token_type_ids=encoded_dict['token_type_ids'],
                                              label=None)
                sub_list_features.append(curr_features)
            input_features.append(sub_list_features)

        return input_features


@register('torch_transformers_re_preprocessor')
class TorchTransformersREPreprocessor(Component):
    def __init__(
            self,
            vocab_file: str,
            special_token: str = '<ENT>',
            ner_tags=None,
            max_seq_length: int = 512,
            do_lower_case: bool = False,
            **kwargs
    ):
        """
        Args:
            vocab_file: path to vocabulary / name of vocabulary for tokenizer initialization
            special_token: an additional token that will be used for marking the entities in the document
            do_lower_case: set True if lowercasing is needed
        Return:
            list of feature batches with input_ids, attention_mask, entity_pos, ner_tags
        """

        self.special_token = special_token
        self.special_tokens_dict = {'additional_special_tokens': [self.special_token]}

        if ner_tags is None:
            ner_tags = ['ORG', 'TIME', 'MISC', 'LOC', 'PER', 'NUM']
        self.ner2id = {tag: tag_id for tag_id, tag in enumerate(ner_tags)}
        self.max_seq_length = max_seq_length

        out = open("log.txt", 'a')
        out.write(f"Number of tags: {len(self.ner2id)}" + '\n')
        out.write(f"Tags are: {self.ner2id}" + '\n')
        out.close()

        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, input_info: List[Tuple[List, List]]) -> List[Dict]:

        tokens, entity_info = zip(*input_info)

        """
        Tokenize and create masks; recalculate the entity positions reagrding the document boarders.
        Args:
            tokens: List of tokens of each document: List[List[tokens in doc]]
            entity_info: List of information about entities containing in the document:
                List[
                    List[
                        (entity1_mention1_start_id, entity1_mention1_end_id),
                        (entity1_mention2_start_id, entity1_mention2_end_id),
                        ],
                    List[
                        (entity2_mention1_start_id, entity2_mention1_end_id),
                        ],
                    NER tag of entity1,
                    NER tag of entity2
                    ]
        Return:
            input_features: List[
                input_ids: List[int],
                attention_mask: List[int],
                entity_pos: List[
                                List[
                                        tuple(entity1_mention1_start_id, entity1_mention1_end_id),
                                        tuple(entity1_mention2_start_id, entity1_mention2_end_id)
                                    ],
                                List[
                                        tuple(entity2_mention1_start_id, entity2_mention1_end_id)
                                    ]
                                ]
                ner_tags: List[ner_tag_entity1, ner_tag_entity2]
                ]
        """

        _ = self.tokenizer.add_special_tokens(self.special_tokens_dict)
        input_features = []
        doc_wordpiece_tokens_batch = []
        entity_pos_batch = []
        ner_tags_batch = []
        for doc, entities in zip(tokens, entity_info):
            count = 0
            doc_wordpiece_tokens = []
            entity1_pos_start = list(zip(*entities[0]))[0]  # first entity mentions' start positions
            entity1_pos_end = list(zip(*entities[0]))[1]  # first entity mentions' end positions
            entity2_pos_start = list(zip(*entities[1]))[0]  # second entity mentions' start positions
            entity2_pos_end = list(zip(*entities[1]))[1]  # second entity mentions' end positions
            upd_entity1_pos_start, upd_entity2_pos_start, upd_entity1_pos_end, upd_entity2_pos_end = [], [], [], []
            for n, token in enumerate(doc):
                if n in entity1_pos_start:
                    doc_wordpiece_tokens.append(self.special_token)
                    upd_entity1_pos_start.append(count)
                    count += 1
                if n in entity1_pos_end:
                    doc_wordpiece_tokens.append(self.special_token)
                    count += 1
                    upd_entity1_pos_end.append(count)

                if n in entity2_pos_start:
                    doc_wordpiece_tokens.append(self.special_token)
                    upd_entity2_pos_start.append(count)
                    count += 1
                if n in entity2_pos_end:
                    doc_wordpiece_tokens.append(self.special_token)
                    count += 1
                    upd_entity2_pos_end.append(count)

                word_tokens = self.tokenizer.tokenize(token)
                doc_wordpiece_tokens += word_tokens
                count += len(word_tokens)

            # special case when the entity is the last in the doc
            if len(doc) in entity1_pos_end:
                doc_wordpiece_tokens.append(self.special_token)
                count += 1
                upd_entity1_pos_end.append(count)
            if len(doc) in entity2_pos_end:
                doc_wordpiece_tokens.append(self.special_token)
                count += 1
                upd_entity2_pos_end.append(count)
                word_tokens = self.tokenizer.tokenize(token)
                doc_wordpiece_tokens += word_tokens
                count += len(word_tokens)

            upd_entity1 = list(zip(upd_entity1_pos_start, upd_entity1_pos_end))
            upd_entity2 = list(zip(upd_entity2_pos_start, upd_entity2_pos_end))

            # text entities for self check
            upd_entity1_text = [doc_wordpiece_tokens[ent_m[0]:ent_m[1]] for ent_m in upd_entity1]
            upd_entity2_text = [doc_wordpiece_tokens[ent_m[0]:ent_m[1]] for ent_m in upd_entity2]

            enc_ner_tag = self.encode_ner_tag(entities[2], entities[3])

            doc_wordpiece_tokens_batch.append(doc_wordpiece_tokens[:self.max_seq_length])   # add truncated tokens
            entity_pos_batch.append([upd_entity1, upd_entity2])
            ner_tags_batch.append(enc_ner_tag)

        for (upd_entity1, upd_entity2), enc_ner_tag, doc_wordpiece_tokens in \
                zip(entity_pos_batch, ner_tags_batch, doc_wordpiece_tokens_batch):
            encoding = self.tokenizer.encode_plus(
                doc_wordpiece_tokens,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_seq_length,
                pad_to_max_length=True,
                return_attention_mask=True,  # return_tensors="pt"
            )

            input_features.append(
                {
                    "input_ids": encoding['input_ids'],
                    "attention_mask": encoding['attention_mask'],
                    "entity_pos": [upd_entity1, upd_entity2],
                    "ner_tags": enc_ner_tag
                }
            )

        # todo: wil be deleted
        # from joblib import dump
        # dump(input_features[:50],
        #      "/Users/asedova/Documents/04_deeppavlov/deeppavlov_fork/DocRED/out_transformer_preprocessor/dev_small")

        return input_features

    def encode_ner_tag(self, *ner_tags) -> List:
        """ Encode NER tags with one hot encodings """
        enc_ner_tags = []
        for ner_tag in ner_tags:
            ner_tag_one_hot = [0] * len(self.ner2id)
            ner_tag_one_hot[self.ner2id[ner_tag]] = 1
            enc_ner_tags.append(ner_tag_one_hot)
        return enc_ner_tags


# todo: wil be deleted
if __name__ == "__main__":
    from joblib import load
    from deeppavlov.dataset_iterators.basic_classification_iterator import BasicClassificationDatasetIterator

    data = load(
        "/Users/asedova/PycharmProjects/05_deeppavlov_fork/docred/out_dataset_reader_without_neg/all_data"
    )
    data_iter_out = BasicClassificationDatasetIterator(data)
    entity_info = [data[0] for data in data_iter_out.test]
    labels = [data[1] for data in data_iter_out.test]

    TorchTransformersREPreprocessor("bert-base-cased").__call__(entity_info)
