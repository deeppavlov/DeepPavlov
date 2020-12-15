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
from logging import getLogger
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from overrides import overrides
from transformers import BertForNextSentencePrediction, BertConfig

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.preprocessors.torch_transformers_preprocessor import TorchTransformersPreprocessor

logger = getLogger(__name__)


@register('torch_bert_as_summarizer')
class TorchBertAsSummarizer(TorchModel):
    """Naive Extractive Summarization model based on BERT on PyTorch.
    BERT model was trained on Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) tasks.
    NSP head was trained to detect in ``[CLS] text_a [SEP] text_b [SEP]`` if text_b follows text_a in original document.

    This NSP head can be used to stack sentences from a long document, based on a initial sentence:

    summary_0 = init_sentence

    summary_1 = summary_0 + argmax(nsp_score(candidates))

    summary_2 = summary_1 + argmax(nsp_score(candidates))

    ...

    , where candidates are all sentences from a document.

    Args:
        pretrained_bert: pretrained Bert checkpoint path or key title (e.g. "bert-base-uncased")
        bert_config_file: path to Bert configuration file (not used if pretrained_bert is key title)
        vocab_file: path to Bert vocabulary
        max_summary_length: limit on summary length, number of sentences is used if ``max_summary_length_in_tokens``
            is set to False, else number of tokens is used.
        max_summary_length_in_tokens: Use number of tokens as length of summary.
            Defaults to ``False``.
        max_seq_length: max sequence length in subtokens, including ``[SEP]`` and ``[CLS]`` tokens.
            `max_seq_length` is used in Bert to compute NSP scores. Defaults to ``128``.
        do_lower_case: set ``True`` if lowercasing is needed. Defaults to ``False``.
        lang: use ru_sent_tokenizer for 'ru' and ntlk.sent_tokener for other languages.
            Defaults to ``'ru'``.
    """

    def __init__(self, pretrained_bert: str,
                 vocab_file: str,
                 max_summary_length: int,
                 bert_config_file: Optional[str] = None,
                 max_summary_length_in_tokens: bool = False,
                 max_seq_length: int = 128,
                 do_lower_case: bool = False,
                 lang: str = 'ru',
                 save_path: Optional[str] = None,
                 **kwargs) -> None:

        self.max_summary_length = max_summary_length
        self.max_summary_length_in_tokens = max_summary_length_in_tokens
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.bert_preprocessor = TorchTransformersPreprocessor(vocab_file=vocab_file, do_lower_case=do_lower_case,
                                                       max_seq_length=max_seq_length)

        self.tokenize_reg = re.compile(r"[\w']+|[^\w ]")

        if lang == 'ru':
            from ru_sent_tokenize import ru_sent_tokenize
            self.sent_tokenizer = ru_sent_tokenize
        else:
            from nltk import sent_tokenize
            self.sent_tokenizer = sent_tokenize

        super().__init__(save_path=save_path, **kwargs)

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_bert and not Path(self.pretrained_bert).is_file():
            self.model = BertForNextSentencePrediction.from_pretrained(
                self.pretrained_bert, output_attentions=False, output_hidden_states=False)
        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.bert_config = BertConfig.from_json_file(str(expand_path(self.bert_config_file)))

            if self.attention_probs_keep_prob is not None:
                self.bert_config.attention_probs_dropout_prob = 1.0 - self.attention_probs_keep_prob
            if self.hidden_keep_prob is not None:
                self.bert_config.hidden_dropout_prob = 1.0 - self.hidden_keep_prob
            self.model = BertForNextSentencePrediction(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        self.model.to(self.device)

    def _get_nsp_predictions(self, sentences: List[str], candidates: List[str]):
        """Compute NextSentence probability for every (sentence_i, candidate_i) pair.

        [CLS] sentence_i [SEP] candidate_i [SEP]

        Args:
            sentences: list of sentences
            candidates: list of candidates to be the next sentence

        Returns:
            probabilities that candidate is a next sentence
        """
        features = self.bert_preprocessor(texts_a=sentences, texts_b=candidates)

        input_ids = [f.input_ids for f in features]
        input_masks = [f.attention_mask for f in features]
        input_type_ids = [f.token_type_ids for f in features]

        b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
        b_input_masks = torch.cat(input_masks, dim=0).to(self.device)
        b_input_type_ids = torch.cat(input_type_ids, dim=0).to(self.device)

        pred = self.model(input_ids=b_input_ids, attention_mask=b_input_masks, token_type_ids=b_input_type_ids)[0]
        nsp_probs = torch.nn.functional.softmax(pred, dim=-1)
        return nsp_probs[:, 0]

    def __call__(self, texts: List[str], init_sentences: Optional[List[str]] = None) -> List[List[str]]:
        """Builds summary for text from `texts`

        Args:
            texts: texts to build summaries for
            init_sentences: ``init_sentence`` is used as the first sentence in summary.
                Defaults to None.

        Returns:
            List[List[str]]: summaries tokenized on sentences
        """
        summaries = []
        # build summaries for each text, init_sentence pair
        if init_sentences is None:
            init_sentences = [None] * len(texts)

        for text, init_sentence in zip(texts, init_sentences):
            text_sentences = self.sent_tokenizer(text)

            if init_sentence is None:
                init_sentence = text_sentences[0]
                text_sentences = text_sentences[1:]

            # remove duplicates
            text_sentences = list(set(text_sentences))
            # remove init_sentence from text sentences
            text_sentences = [sent for sent in text_sentences if sent != init_sentence]

            summary = [init_sentence]
            if self.max_summary_length_in_tokens:
                # get length in tokens
                def get_length(x):
                    return len(self.tokenize_reg.findall(' '.join(x)))
            else:
                # get length as number of sentences
                get_length = len

            candidates = text_sentences[:]
            while len(candidates) > 0:
                # todo: use batches
                candidates_scores = [self._get_nsp_predictions([' '.join(summary)], [cand]) for cand in candidates]
                best_candidate_idx = np.argmax(candidates_scores)
                best_candidate = candidates[best_candidate_idx]
                del candidates[best_candidate_idx]
                if get_length(summary + [best_candidate]) > self.max_summary_length:
                    break
                summary = summary + [best_candidate]
            summaries += [summary]
        return summaries

    def train_on_batch(self, **kwargs):
        raise NotImplementedError
