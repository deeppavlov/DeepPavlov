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

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Set, Any

import nltk
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.data.processors.utils import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component
from deeppavlov.models.preprocessors.mask import Mask

log = getLogger(__name__)


@register('torch_transformers_multiplechoice_preprocessor')
class TorchTransformersMultiplechoicePreprocessor(Component):
    """Tokenize text on subtokens, encode subtokens with their indices, create tokens and segment masks.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        tokenizer: instance of Bert FullTokenizer

    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def tokenize_mc_examples(self,
                             contexts: List[List[str]],
                             choices: List[List[str]]) -> Dict[str, torch.tensor]:

        num_choices = len(contexts[0])
        batch_size = len(contexts)

        # tokenize examples in groups of `num_choices`
        examples = []
        for context_list, choice_list in zip(contexts, choices):
            for context, choice in zip(context_list, choice_list):
                tokenized_input = self.tokenizer.encode_plus(text=context,
                                                             text_pair=choice,
                                                             return_attention_mask=True,
                                                             add_special_tokens=True,
                                                             truncation=True)

                examples.append(tokenized_input)

        padded_examples = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=self.max_seq_length,
            return_tensors='pt',
        )

        padded_examples = {k: v.view(batch_size, num_choices, -1) for k, v in padded_examples.items()}

        return padded_examples

    def __call__(self, texts_a: List[List[str]], texts_b: List[List[str]] = None) -> Dict[str, torch.tensor]:
        """Tokenize and create masks.

        texts_a and texts_b are separated by [SEP] token

        Args:
            texts_a: list of texts,
            texts_b: list of texts, it could be None, e.g. single sentence classification task

        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
        """

        input_features = self.tokenize_mc_examples(texts_a, texts_b)
        return input_features


@register('torch_transformers_preprocessor')
class TorchTransformersPreprocessor(Component):
    """Tokenize text on subtokens, encode subtokens with their indices, create tokens and segment masks.

    Args:
        vocab_file: A string, the `model id` of a predefined tokenizer hosted inside a model repo on huggingface.co or
            a path to a `directory` containing vocabulary files required by the tokenizer.
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        tokenizer: instance of Bert FullTokenizer

    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, texts_a: List[str], texts_b: Optional[List[str]] = None) -> Union[List[InputFeatures],
                                                                                         Tuple[List[InputFeatures],
                                                                                               List[List[str]]]]:
        """Tokenize and create masks.
        texts_a and texts_b are separated by [SEP] token
        Args:
            texts_a: list of texts,
            texts_b: list of texts, it could be None, e.g. single sentence classification task
        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
        """

        # in case of iterator's strange behaviour
        if isinstance(texts_a, tuple):
            texts_a = list(texts_a)

        input_features = self.tokenizer(text=texts_a,
                                        text_pair=texts_b,
                                        add_special_tokens=True,
                                        max_length=self.max_seq_length,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        truncation=True,
                                        return_tensors='pt')
        return input_features


@register('torch_transformers_entity_ranker_preprocessor')
class TorchTransformersEntityRankerPreprocessor(Component):
    """Class for tokenization of text into subtokens, encoding of subtokens with indices and obtaining positions of
    special [ENT]-tokens
    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        special_tokens: list of special tokens
        special_token_id: id of special token
        return_special_tokens_pos: whether to return positions of found special tokens
    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = False,
                 max_seq_length: int = 512,
                 special_tokens: List[str] = None,
                 special_token_id: int = None,
                 return_special_tokens_pos: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        if special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.special_token_id = special_token_id
        self.return_special_tokens_pos = return_special_tokens_pos

    def __call__(self, texts_a: List[str]) -> Tuple[Any, List[int]]:
        """Tokenize and find special tokens positions.
        Args:
            texts_a: list of texts,
        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
            batch of indices of special token ids in input ids sequence
        """
        # in case of iterator's strange behaviour
        if isinstance(texts_a, tuple):
            texts_a = list(texts_a)
        if self.do_lower_case:
            texts_a = [text.lower() for text in texts_a]
        lengths = []
        input_ids_batch = []
        for text_a in texts_a:
            encoding = self.tokenizer.encode_plus(
                text_a, add_special_tokens=True, pad_to_max_length=True, return_attention_mask=True)
            input_ids = encoding["input_ids"]
            input_ids_batch.append(input_ids)
            lengths.append(len(input_ids))

        max_length = min(max(lengths), self.max_seq_length)
        input_features = self.tokenizer(text=texts_a,
                                        add_special_tokens=True,
                                        max_length=max_length,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        truncation=True,
                                        return_tensors='pt')
        special_tokens_pos = []
        for input_ids_list in input_ids_batch:
            found_n = -1
            for n, input_id in enumerate(input_ids_list):
                if input_id == self.special_token_id:
                    found_n = n
                    break
            if found_n == -1:
                found_n = 0
            special_tokens_pos.append(found_n)

        if self.return_special_tokens_pos:
            return input_features, special_tokens_pos
        else:
            return input_features


@register('torch_squad_transformers_preprocessor')
class TorchSquadTransformersPreprocessor(Component):
    """Tokenize text on subtokens, encode subtokens with their indices, create tokens and segment masks.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        tokenizer: instance of Bert FullTokenizer

    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 add_token_type_ids: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.add_token_type_ids = add_token_type_ids
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, question_batch: List[str], context_batch: Optional[List[str]] = None) -> Union[
        List[InputFeatures],
        Tuple[List[InputFeatures],
              List[List[str]]]]:
        """Tokenize and create masks.

        texts_a_batch and texts_b_batch are separated by [SEP] token

        Args:
            texts_a_batch: list of texts,
            texts_b_batch: list of texts, it could be None, e.g. single sentence classification task

        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures, batch of subtokens and batch of
                split paragraphs
        """

        if context_batch is None:
            context_batch = [None] * len(question_batch)

        input_features_batch, tokens_batch, split_context_batch = [], [], []
        for question, context in zip(question_batch, context_batch):
            question_list, context_list = [], []
            context_subtokens = self.tokenizer.tokenize(context)
            question_subtokens = self.tokenizer.tokenize(question)
            max_chunk_len = self.max_seq_length - len(question_subtokens) - 3
            if 0 < max_chunk_len < len(context_subtokens):
                number_of_chunks = math.ceil(len(context_subtokens) / max_chunk_len)
                sentences = nltk.sent_tokenize(context)
                for chunk in np.array_split(sentences, number_of_chunks):
                    context_list += [' '.join(chunk)]
                    question_list += [question]
            else:
                context_list += [context]
                question_list += [question]

            input_features_list, tokens_list = [], []
            for question_elem, context_elem in zip(question_list, context_list):
                encoded_dict = self.tokenizer.encode_plus(
                    text=question_elem, text_pair=context_elem,
                    add_special_tokens=True,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt')
                if 'token_type_ids' not in encoded_dict:
                    if self.add_token_type_ids:
                        input_ids = encoded_dict['input_ids']
                        seq_len = input_ids.size(1)
                        sep = torch.where(input_ids == self.tokenizer.sep_token_id)[1][0].item()
                        len_a = min(sep + 1, seq_len)
                        len_b = seq_len - len_a
                        encoded_dict['token_type_ids'] = torch.cat((torch.zeros(1, len_a, dtype=int),
                                                                    torch.ones(1, len_b, dtype=int)), dim=1)
                    else:
                        encoded_dict['token_type_ids'] = torch.tensor([0])

                curr_features = InputFeatures(input_ids=encoded_dict['input_ids'],
                                              attention_mask=encoded_dict['attention_mask'],
                                              token_type_ids=encoded_dict['token_type_ids'],
                                              label=None)
                input_features_list.append(curr_features)
                tokens_list.append(self.tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0]))

            input_features_batch.append(input_features_list)
            tokens_batch.append(tokens_list)
            split_context_batch.append(context_list)

        return input_features_batch, tokens_batch, split_context_batch


@register('rel_ranking_preprocessor')
class RelRankingPreprocessor(Component):
    """Class for tokenization of text and relation labels
    Args:
        vocab_file: path to vocabulary
        add_special_tokens: special_tokens_list
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
    """

    def __init__(self,
                 vocab_file: str,
                 add_special_tokens: List[str],
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        self.add_special_tokens = add_special_tokens
        special_tokens_dict = {'additional_special_tokens': add_special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)

    def __call__(self, questions_batch: List[List[str]], rels_batch: List[List[str]] = None) -> Dict[str, torch.tensor]:
        """Tokenize questions and relations
        texts_a and texts_b are separated by [SEP] token
        Args:
            questions_batch: list of texts,
            rels_batch: list of relations list

        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
        """
        lengths = []
        for question, rels_list in zip(questions_batch, rels_batch):
            if isinstance(rels_list, list):
                rels_str = self.add_special_tokens[2].join(rels_list)
            else:
                rels_str = rels_list
            text_input = f"{self.add_special_tokens[0]} {question} {self.add_special_tokens[1]} {rels_str}"
            encoding = self.tokenizer.encode_plus(text=text_input,
                                                  return_attention_mask=True, add_special_tokens=True,
                                                  truncation=True)
            lengths.append(len(encoding["input_ids"]))
        max_len = max(lengths)
        input_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []
        for question, rels_list in zip(questions_batch, rels_batch):
            if isinstance(rels_list, list):
                rels_str = self.add_special_tokens[2].join(rels_list)
            else:
                rels_str = rels_list
            text_input = f"{self.add_special_tokens[0]} {question} {self.add_special_tokens[1]} {rels_str}"
            encoding = self.tokenizer.encode_plus(text=text_input,
                                                  truncation = True, max_length=max_len,
                                                  pad_to_max_length=True, return_attention_mask = True)
            input_ids_batch.append(encoding["input_ids"])
            attention_mask_batch.append(encoding["attention_mask"])
            if "token_type_ids" in encoding:
                token_type_ids_batch.append(encoding["token_type_ids"])
            else:
                token_type_ids_batch.append([0])
            
        input_features = {"input_ids": torch.LongTensor(input_ids_batch),
                          "attention_mask": torch.LongTensor(attention_mask_batch),
                          "token_type_ids": torch.LongTensor(token_type_ids_batch)}
            
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
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
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


@dataclass
class RecordFlatExample:
    """Dataclass to store a flattened ReCoRD example. Contains `probability` for
    a given `entity` candidate, as well as its label.
    """
    index: str
    label: int
    probability: float
    entity: str


@dataclass
class RecordNestedExample:
    """Dataclass to store a nested ReCoRD example. Contains a single predicted entity, as well as
    a list of correct answers.
    """
    index: str
    prediction: str
    answers: List[str]


@register("torch_record_postprocessor")
class TorchRecordPostprocessor:
    """Combines flat classification examples into nested examples. When called returns nested examples
    that weren't previously returned during current iteration over examples.

    Args:
        is_binary: signifies whether the classifier uses binary classification head
    Attributes:
        record_example_accumulator: underling accumulator that transforms flat examples
        total_examples: overall number of flat examples that must be processed during current iteration
    """

    def __init__(self, is_binary: bool = False, *args, **kwargs):
        self.record_example_accumulator: RecordExampleAccumulator = RecordExampleAccumulator()
        self.total_examples: Optional[int, None] = None
        self.is_binary: bool = is_binary

    def __call__(self,
                 idx: List[str],
                 y: List[int],
                 y_pred_probas: np.ndarray,
                 entities: List[str],
                 num_examples: List[int],
                 *args,
                 **kwargs) -> List[RecordNestedExample]:
        """Postprocessor call

        Args:
            idx: list of string indices
            y: list of integer labels
            y_pred_probas: array of predicted probabilities
            num_examples: list of duplicated total numbers of examples

        Returns:
            List[RecordNestedExample]: processed but not previously returned examples (may be empty in some cases)
        """
        if not self.is_binary:
            # if we have outputs for both classes `0` and `1`
            y_pred_probas = y_pred_probas[:, 1]
        if self.total_examples != num_examples[0]:
            # start over if num_examples is different
            # implying that a different split is being evaluated
            self.reset_accumulator()
            self.total_examples = num_examples[0]
        for index, label, probability, entity in zip(idx, y, y_pred_probas, entities):
            self.record_example_accumulator.add_flat_example(index, label, probability, entity)
            self.record_example_accumulator.collect_nested_example(index)
            if self.record_example_accumulator.examples_processed >= self.total_examples:
                # start over if all examples were processed
                self.reset_accumulator()
        return self.record_example_accumulator.return_examples()

    def reset_accumulator(self):
        """Reinitialize the underlying accumulator from scratch
        """
        self.record_example_accumulator = RecordExampleAccumulator()


class RecordExampleAccumulator:
    """ReCoRD example accumulator

    Attributes:
        examples_processed: total number of examples processed so far
        record_counter: number of examples processed for each index
        nested_len: expected number of flat examples for a given index
        flat_examples: stores flat examples
        nested_examples: stores nested examples
        collected_indices: indices of collected nested examples
        returned_indices: indices that have been returned
    """

    def __init__(self):
        self.examples_processed: int = 0
        self.record_counter: Dict[str, int] = defaultdict(lambda: 0)
        self.nested_len: Dict[str, int] = dict()
        self.flat_examples: Dict[str, List[RecordFlatExample]] = defaultdict(lambda: [])
        self.nested_examples: Dict[str, RecordNestedExample] = dict()
        self.collected_indices: Set[str] = set()
        self.returned_indices: Set[str] = set()

    def add_flat_example(self, index: str, label: int, probability: float, entity: str):
        """Add a single flat example to the accumulator

        Args:
            index: example index
            label: example label (`-1` means that label is not available)
            probability: predicted probability
            entity: candidate entity
        """
        self.flat_examples[index].append(RecordFlatExample(index, label, probability, entity))
        if index not in self.nested_len:
            self.nested_len[index] = self.get_expected_len(index)
        self.record_counter[index] += 1
        self.examples_processed += 1

    def ready_to_nest(self, index: str) -> bool:
        """Checks whether all the flat examples for a given index were collected at this point.
        Args:
            index: the index of the candidate nested example
        Returns:
            bool: indicates whether the collected flat examples can be combined into a nested example
        """
        return self.record_counter[index] == self.nested_len[index]

    def collect_nested_example(self, index: str):
        """Combines a list of flat examples denoted by the given index into a single nested example
        provided that all the necessary flat example have been collected by this time.
        Args:
            index: the index of the candidate nested example
        """
        if self.ready_to_nest(index):
            example_list: List[RecordFlatExample] = self.flat_examples[index]
            entities: List[str] = []
            labels: List[int] = []
            probabilities: List[float] = []
            answers: List[str] = []

            for example in example_list:
                entities.append(example.entity)
                labels.append(example.label)
                probabilities.append(example.probability)
                if example.label == 1:
                    answers.append(example.entity)

            prediction_index = np.argmax(probabilities)
            prediction = entities[prediction_index]

            self.nested_examples[index] = RecordNestedExample(index, prediction, answers)
            self.collected_indices.add(index)

    def return_examples(self) -> List[RecordNestedExample]:
        """Determines which nested example were not yet returned during the current evaluation
        cycle and returns them. May return an empty list if there are no new nested examples
        to return yet.
        Returns:
            List[RecordNestedExample]: zero or more nested examples
        """
        indices_to_return: Set[str] = self.collected_indices.difference(self.returned_indices)
        examples_to_return: List[RecordNestedExample] = []
        for index in indices_to_return:
            examples_to_return.append(self.nested_examples[index])
        self.returned_indices.update(indices_to_return)
        return examples_to_return

    @staticmethod
    def get_expected_len(index: str) -> int:
        """
        Calculates the total number of flat examples denoted by the give index
        Args:
            index: the index to calculate the number of examples for
        Returns:
            int: the expected number of examples for this index
        """
        return int(index.split("-")[-1])
