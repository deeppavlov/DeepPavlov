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
from typing import List, Optional

import numpy as np
import tensorflow.compat.v1 as tf

from bert_dp.modeling import BertConfig, BertModel, create_initializer, get_assignment_map_from_checkpoint
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor

logger = getLogger(__name__)


@register('bert_as_summarizer')
class BertAsSummarizer(TFModel):
    """Naive Extractive Summarization model based on BERT.
    BERT model was trained on Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) tasks.
    NSP head was trained to detect in ``[CLS] text_a [SEP] text_b [SEP]`` if text_b follows text_a in original document.

    This NSP head can be used to stack sentences from a long document, based on a initial sentence:

    summary_0 = init_sentence

    summary_1 = summary_0 + argmax(nsp_score(candidates))

    summary_2 = summary_1 + argmax(nsp_score(candidates))

    ...

    , where candidates are all sentences from a document.

    Args:
        bert_config_file: path to Bert configuration file
        pretrained_bert: path to pretrained Bert checkpoint
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

    def __init__(self, bert_config_file: str,
                 pretrained_bert: str,
                 vocab_file: str,
                 max_summary_length: int,
                 max_summary_length_in_tokens: Optional[bool] = False,
                 max_seq_length: Optional[int] = 128,
                 do_lower_case: Optional[bool] = False,
                 lang: Optional[str] = 'ru',
                 **kwargs) -> None:

        self.max_summary_length = max_summary_length
        self.max_summary_length_in_tokens = max_summary_length_in_tokens
        self.bert_config = BertConfig.from_json_file(str(expand_path(bert_config_file)))

        self.bert_preprocessor = BertPreprocessor(vocab_file=vocab_file, do_lower_case=do_lower_case,
                                                  max_seq_length=max_seq_length)

        self.tokenize_reg = re.compile(r"[\w']+|[^\w ]")

        if lang == 'ru':
            from ru_sent_tokenize import ru_sent_tokenize
            self.sent_tokenizer = ru_sent_tokenize
        else:
            from nltk import sent_tokenize
            self.sent_tokenizer = sent_tokenize

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self._init_graph()

        self.sess.run(tf.global_variables_initializer())

        if pretrained_bert is not None:
            pretrained_bert = str(expand_path(pretrained_bert))

            if tf.train.checkpoint_exists(pretrained_bert):
                logger.info('[initializing model with Bert from {}]'.format(pretrained_bert))
                tvars = tf.trainable_variables()
                assignment_map, _ = get_assignment_map_from_checkpoint(tvars, pretrained_bert)
                tf.train.init_from_checkpoint(pretrained_bert, assignment_map)

    def _init_graph(self):
        self._init_placeholders()

        self.bert = BertModel(config=self.bert_config,
                              is_training=self.is_train_ph,
                              input_ids=self.input_ids_ph,
                              input_mask=self.input_masks_ph,
                              token_type_ids=self.token_types_ph,
                              use_one_hot_embeddings=False,
                              )
        # next sentence prediction head
        with tf.variable_scope("cls/seq_relationship"):
            output_weights = tf.get_variable(
                "output_weights",
                shape=[2, self.bert_config.hidden_size],
                initializer=create_initializer(self.bert_config.initializer_range))
            output_bias = tf.get_variable(
                "output_bias", shape=[2], initializer=tf.zeros_initializer())

        nsp_logits = tf.matmul(self.bert.get_pooled_output(), output_weights, transpose_b=True)
        nsp_logits = tf.nn.bias_add(nsp_logits, output_bias)
        self.nsp_probs = tf.nn.softmax(nsp_logits, axis=-1)

    def _init_placeholders(self):
        self.input_ids_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ids_ph')
        self.input_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='masks_ph')
        self.token_types_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_types_ph')

        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _build_feed_dict(self, input_ids, input_masks, token_types):
        feed_dict = {
            self.input_ids_ph: input_ids,
            self.input_masks_ph: input_masks,
            self.token_types_ph: token_types,
        }
        return feed_dict

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
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]
        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids)
        nsp_probs = self.sess.run(self.nsp_probs, feed_dict=feed_dict)
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
