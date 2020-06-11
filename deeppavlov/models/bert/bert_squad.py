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

import json
import math
from logging import getLogger
from typing import List, Tuple, Optional, Dict

import numpy as np
import tensorflow as tf
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer
from bert_dp.preprocessing import InputFeatures
from bert_dp.tokenization import FullTokenizer

from deeppavlov import build_model
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.models.squad.utils import softmax_mask

logger = getLogger(__name__)


@register('squad_bert_model')
class BertSQuADModel(LRScheduledTFModel):
    """Bert-based model for SQuAD-like problem setting:
    It predicts start and end position of answer for given question and context.

    [CLS] token is used as no_answer. If model selects [CLS] token as most probable
    answer, it means that there is no answer in given context.

    Start and end position of answer are predicted by linear transformation
    of Bert outputs.

    Args:
        bert_config_file: path to Bert configuration file
        keep_prob: dropout keep_prob for non-Bert layers
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        optimizer: name of tf.train.* optimizer or None for `AdamWeightDecayOptimizer`
        weight_decay_rate: L2 weight decay for `AdamWeightDecayOptimizer`
        pretrained_bert: pretrained Bert checkpoint
        min_learning_rate: min value of learning rate if learning rate decay is used
    """

    def __init__(self, bert_config_file: str,
                 keep_prob: float,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 optimizer: Optional[str] = None,
                 weight_decay_rate: Optional[float] = 0.01,
                 pretrained_bert: Optional[str] = None,
                 min_learning_rate: float = 1e-06, **kwargs) -> None:
        super().__init__(**kwargs)

        self.min_learning_rate = min_learning_rate
        self.keep_prob = keep_prob
        self.optimizer = optimizer
        self.weight_decay_rate = weight_decay_rate

        self.bert_config = BertConfig.from_json_file(str(expand_path(bert_config_file)))

        if attention_probs_keep_prob is not None:
            self.bert_config.attention_probs_dropout_prob = 1.0 - attention_probs_keep_prob
        if hidden_keep_prob is not None:
            self.bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self._init_graph()

        self._init_optimizer()

        self.sess.run(tf.global_variables_initializer())

        if pretrained_bert is not None:
            pretrained_bert = str(expand_path(pretrained_bert))

            if tf.train.checkpoint_exists(pretrained_bert) \
                    and not (self.load_path and tf.train.checkpoint_exists(str(self.load_path.resolve()))):
                logger.info('[initializing model with Bert from {}]'.format(pretrained_bert))
                var_list = self._get_saveable_variables(
                    exclude_scopes=('Optimizer', 'learning_rate', 'momentum', 'squad'))
                saver = tf.train.Saver(var_list)
                saver.restore(self.sess, pretrained_bert)

        if self.load_path is not None:
            self.load()

    def _init_graph(self):
        self._init_placeholders()

        seq_len = tf.shape(self.input_ids_ph)[-1]
        self.y_st = tf.one_hot(self.y_st_ph, depth=seq_len)
        self.y_end = tf.one_hot(self.y_end_ph, depth=seq_len)

        self.bert = BertModel(config=self.bert_config,
                              is_training=self.is_train_ph,
                              input_ids=self.input_ids_ph,
                              input_mask=self.input_masks_ph,
                              token_type_ids=self.token_types_ph,
                              use_one_hot_embeddings=False,
                              )

        last_layer = self.bert.get_sequence_output()
        hidden_size = last_layer.get_shape().as_list()[-1]
        bs = tf.shape(last_layer)[0]

        with tf.variable_scope('squad'):
            output_weights = tf.get_variable('output_weights', [2, hidden_size],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_bias = tf.get_variable('output_bias', [2], initializer=tf.zeros_initializer())

            last_layer_rs = tf.reshape(last_layer, [-1, hidden_size])

            logits = tf.matmul(last_layer_rs, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [bs, -1, 2])
            logits = tf.transpose(logits, [2, 0, 1])

            logits_st, logits_end = tf.unstack(logits, axis=0)

            logit_mask = self.token_types_ph
            # [CLS] token is used as no answer
            mask = tf.concat([tf.ones((bs, 1), dtype=tf.int32), tf.zeros((bs, seq_len - 1), dtype=tf.int32)], axis=-1)
            logit_mask = logit_mask + mask

            logits_st = softmax_mask(logits_st, logit_mask)
            logits_end = softmax_mask(logits_end, logit_mask)
            start_probs = tf.nn.softmax(logits_st)
            end_probs = tf.nn.softmax(logits_end)

            outer = tf.matmul(tf.expand_dims(start_probs, axis=2), tf.expand_dims(end_probs, axis=1))
            outer_logits = tf.exp(tf.expand_dims(logits_st, axis=2) + tf.expand_dims(logits_end, axis=1))

            context_max_len = tf.reduce_max(tf.reduce_sum(self.token_types_ph, axis=1))

            max_ans_length = tf.cast(tf.minimum(20, context_max_len), tf.int64)
            outer = tf.matrix_band_part(outer, 0, max_ans_length)
            outer_logits = tf.matrix_band_part(outer_logits, 0, max_ans_length)

            self.yp_score = 1 - tf.nn.softmax(logits_st)[:, 0] * tf.nn.softmax(logits_end)[:, 0]

            self.start_probs = start_probs
            self.end_probs = end_probs
            self.start_pred = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.end_pred = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            self.yp_logits = tf.reduce_max(tf.reduce_max(outer_logits, axis=2), axis=1)

        with tf.variable_scope("loss"):
            loss_st = tf.nn.softmax_cross_entropy_with_logits(logits=logits_st, labels=self.y_st)
            loss_end = tf.nn.softmax_cross_entropy_with_logits(logits=logits_end, labels=self.y_end)
            self.loss = tf.reduce_mean(loss_st + loss_end)

    def _init_placeholders(self):
        self.input_ids_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='ids_ph')
        self.input_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='masks_ph')
        self.token_types_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_types_ph')

        self.y_st_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='y_st_ph')
        self.y_end_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='y_end_ph')

        self.learning_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='learning_rate_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _init_optimizer(self):
        with tf.variable_scope('Optimizer'):
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            # default optimizer for Bert is Adam with fixed L2 regularization
            if self.optimizer is None:

                self.train_op = self.get_train_op(self.loss, learning_rate=self.learning_rate_ph,
                                                  optimizer=AdamWeightDecayOptimizer,
                                                  weight_decay_rate=self.weight_decay_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=1e-6,
                                                  exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
                                                  )
            else:
                self.train_op = self.get_train_op(self.loss, learning_rate=self.learning_rate_ph)

            if self.optimizer is None:
                new_global_step = self.global_step + 1
                self.train_op = tf.group(self.train_op, [self.global_step.assign(new_global_step)])

    def _build_feed_dict(self, input_ids, input_masks, token_types, y_st=None, y_end=None):
        feed_dict = {
            self.input_ids_ph: input_ids,
            self.input_masks_ph: input_masks,
            self.token_types_ph: token_types,
        }
        if y_st is not None and y_end is not None:
            feed_dict.update({
                self.y_st_ph: y_st,
                self.y_end_ph: y_end,
                self.learning_rate_ph: max(self.get_learning_rate(), self.min_learning_rate),
                self.keep_prob_ph: self.keep_prob,
                self.is_train_ph: True,
            })

        return feed_dict

    def train_on_batch(self, features: List[InputFeatures], y_st: List[List[int]], y_end: List[List[int]]) -> Dict:
        """Train model on given batch.
        This method calls train_op using features and labels from y_st and y_end

        Args:
            features: batch of InputFeatures instances
            y_st: batch of lists of ground truth answer start positions
            y_end: batch of lists of ground truth answer end positions

        Returns:
            dict with loss and learning_rate values

        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]

        y_st = [x[0] for x in y_st]
        y_end = [x[0] for x in y_end]

        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids, y_st, y_end)

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return {'loss': loss, 'learning_rate': feed_dict[self.learning_rate_ph]}

    def __call__(self, features: List[InputFeatures]) -> Tuple[List[int], List[int], List[float], List[float]]:
        """get predictions using features as input

        Args:
            features: batch of InputFeatures instances

        Returns:
            predictions: start, end positions, logits for answer and no_answer score

        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]

        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids)
        st, end, logits, scores = self.sess.run([self.start_pred, self.end_pred, self.yp_logits, self.yp_score],
                                                feed_dict=feed_dict)
        return st, end, logits.tolist(), scores.tolist()


@register('squad_bert_infer')
class BertSQuADInferModel(Component):
    """This model wraps BertSQuADModel to make predictions on longer than 512 tokens sequences.

    It splits context on chunks with `max_seq_length - 3 - len(question)` length, preserving sentences boundaries.

    It reassembles batches with chunks instead of full contexts to optimize performance, e.g.,:
        batch_size = 5
        number_of_contexts == 2
        number of first context chunks == 8
        number of second context chunks == 2

        we will create two batches with 5 chunks

    For each context the best answer is selected via logits or scores from BertSQuADModel.


    Args:
        squad_model_config: path to DeepPavlov BertSQuADModel config file
        vocab_file: path to Bert vocab file
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        batch_size: size of batch to use during inference
        lang: either `en` or `ru`, it is used to select sentence tokenizer

    """

    def __init__(self, squad_model_config: str,
                 vocab_file: str,
                 do_lower_case: bool,
                 max_seq_length: int = 512,
                 batch_size: int = 10,
                 lang='en', **kwargs) -> None:
        config = json.load(open(squad_model_config))
        config['chainer']['pipe'][0]['max_seq_length'] = max_seq_length
        self.model = build_model(config)
        self.max_seq_length = max_seq_length
        vocab_file = str(expand_path(vocab_file))
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.batch_size = batch_size

        if lang == 'en':
            from nltk import sent_tokenize
            self.sent_tokenizer = sent_tokenize
        elif lang == 'ru':
            from ru_sent_tokenize import ru_sent_tokenize
            self.sent_tokenizer = ru_sent_tokenize
        else:
            raise RuntimeError('en and ru languages are supported only')

    def __call__(self, contexts: List[str], questions: List[str], **kwargs) -> Tuple[List[str], List[int], List[float]]:
        """get predictions for given contexts and questions

        Args:
            contexts: batch of contexts
            questions: batch of questions

        Returns:
            predictions: answer, answer start position, logits or scores

        """
        batch_indices = []
        contexts_to_predict = []
        questions_to_predict = []
        predictions = {}
        for i, (context, question) in enumerate(zip(contexts, questions)):
            context_subtokens = self.tokenizer.tokenize(context)
            question_subtokens = self.tokenizer.tokenize(question)
            max_chunk_len = self.max_seq_length - len(question_subtokens) - 3
            if 0 < max_chunk_len < len(context_subtokens):
                number_of_chunks = math.ceil(len(context_subtokens) / max_chunk_len)
                sentences = self.sent_tokenizer(context)
                for chunk in np.array_split(sentences, number_of_chunks):
                    contexts_to_predict += [' '.join(chunk)]
                    questions_to_predict += [question]
                    batch_indices += [i]
            else:
                contexts_to_predict += [context]
                questions_to_predict += [question]
                batch_indices += [i]

        for j in range(0, len(contexts_to_predict), self.batch_size):
            c_batch = contexts_to_predict[j: j + self.batch_size]
            q_batch = questions_to_predict[j: j + self.batch_size]
            ind_batch = batch_indices[j: j + self.batch_size]
            a_batch, a_st_batch, logits_batch = self.model(c_batch, q_batch)
            for a, a_st, logits, ind in zip(a_batch, a_st_batch, logits_batch, ind_batch):
                if ind in predictions:
                    predictions[ind] += [(a, a_st, logits)]
                else:
                    predictions[ind] = [(a, a_st, logits)]

        answers, answer_starts, logits = [], [], []
        for ind in sorted(predictions.keys()):
            prediction = predictions[ind]
            best_answer_ind = np.argmax([p[2] for p in prediction])
            answers += [prediction[best_answer_ind][0]]
            answer_starts += [prediction[best_answer_ind][1]]
            logits += [prediction[best_answer_ind][2]]

        return answers, answer_starts, logits
