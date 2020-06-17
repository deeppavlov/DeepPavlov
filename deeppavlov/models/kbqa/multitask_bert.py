# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer
from bert_dp.preprocessing import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.layers.tf_layers import bi_rnn
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.models.bert.bert_sequence_tagger import ExponentialMovingAverage, token_from_subtoken

log = getLogger(__name__)


@register('mt_bert')
class MultiTaskBert(LRScheduledTFModel):
    """
    Basic class for BERT-based sequential architectures.

    Args:
        keep_prob: dropout keep_prob for non-Bert layers
        bert_config_file: path to Bert configuration file
        pretrained_bert: pretrained Bert checkpoint
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        encoder_layer_ids: list of averaged layers from Bert encoder (layer ids)
            optimizer: name of tf.train.* optimizer or None for `AdamWeightDecayOptimizer`
            weight_decay_rate: L2 weight decay for `AdamWeightDecayOptimizer`
        encoder_dropout: dropout probability of encoder output layer
        freeze_embeddings: set True to not train input embeddings set True to
            not train input embeddings set True to not train input embeddings
        learning_rate: learning rate of BERT head
        bert_learning_rate: learning rate of BERT body
        min_learning_rate: min value of learning rate if learning rate decay is used
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        clip_norm: clip gradients by norm
    """
    # TODO: check if ema is really needed and if it can be used on multitask bert
    def __init__(self,
                 shared_params: dict,
                 launches_tasks: dict,
                 inference_launch_names: List[str] = None,
                 **kwargs) -> None:
        super().__init__(learning_rate=shared_params['bert_learning_rate'],
                         learning_rate_drop_div=shared_params['learning_rate_drop_div'],
                         learning_rate_drop_patience=shared_params['learning_rate_drop_patience'],
                         load_before_drop=shared_params['load_before_drop'],
                         clip_norm=shared_params['clip_norm'],
                         **kwargs)
        self.shared_params = shared_params
        self.launches_tasks = launches_tasks
        self.inference_launch_names = inference_launch_names

        self.shared_ph = None  # TODO: add use for `min_body_learning_rate`

        self.bert_config = BertConfig.from_json_file(str(expand_path(self.shared_params['bert_config_file'])))

        if self.shared_params['attention_probs_keep_prob'] is not None:
            self.bert_config.attention_probs_dropout_prob = 1.0 - self.shared_params['attention_probs_keep_prob']
        hidden_keep_prob = self.shared_params.get('hidden_keep_prob')
        if hidden_keep_prob is not None:
            self.bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self._init_bert_body_graph()
        self.tasks = self.build_tasks()

        self.sess.run(tf.global_variables_initializer())

        if self.shared_params['pretrained_bert'] is not None:
            pretrained_bert = str(expand_path(self.shared_params['pretrained_bert']))

            if tf.train.checkpoint_exists(pretrained_bert) \
                    and not (self.load_path and tf.train.checkpoint_exists(str(self.load_path.resolve()))):
                log.info('[initializing model with Bert from {}]'.format(pretrained_bert))
                # Exclude optimizer and classification variables from saved variables
                var_list = self._get_saveable_variables(
                    exclude_scopes=('Optimizer', 'learning_rate', 'momentum', 'ner'))
                saver = tf.train.Saver(var_list)
                saver.restore(self.sess, pretrained_bert)

        if self.load_path is not None:
            self.load()

    def build_tasks(self):
        for launch_name, launch_params in self.launches_tasks.items():
            for task_name, task_obj in launch_params.items():
                task_obj.build(
                    bert_body=self.bert,
                    shared_params=self.shared_params,
                    shared_placeholders=self.shared_ph
                )

    def _init_shared_placeholders(self) -> None:
        # TODO: write comments describing the difference between shared placeholders and shared params
        self.shared_ph = {
            'input_ids': tf.placeholder(shape=(None, None),
                                        dtype=tf.int32,
                                        name='token_indices_ph'),
            'input_masks': tf.placeholder(shape=(None, None),
                                          dtype=tf.int32,
                                          name='token_mask_ph'),
            'token_types': tf.placeholder_with_default(tf.zeros_like(self.input_ids_ph, dtype=tf.int32),
                                                       shape=self.input_ids_ph.shape,
                                                       name='token_types_ph'),
            'learning_rate': tf.placeholder_with_default(0.0, shape=[], name='learning_rate_ph'),
            'keep_prob': tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph'),
            'encoder_keep_prob': tf.placeholder_with_default(1.0, shape=[], name='encoder_keep_prob_ph'),
            'is_train': tf.placeholder_with_default(False, shape=[], name='is_train_ph')}

    def _init_bert_body_graph(self) -> None:
        self._init_shared_placeholders()
        self.seq_lengths = tf.reduce_sum(self.y_masks_ph, axis=1)
        sph = self.shared_ph
        self.bert = BertModel(config=self.bert_config,
                              is_training=sph['is_train'],
                              input_ids=sph['input_ids'],
                              input_mask=sph['input_masks'],
                              token_type_ids=sph['token_types'],
                              use_one_hot_embeddings=False)

    def get_bert_body_train_op(
            self, loss: tf.Tensor, learning_rate: Union[tf.Tensor, float], **kwargs) -> tf.Operation:
        assert "learnable_scopes" not in kwargs, "learnable scopes unsupported"
        # train_op for bert variables
        kwargs['learnable_scopes'] = ('bert/encoder', 'bert/embeddings')
        if self.freeze_embeddings:
            kwargs['learnable_scopes'] = ('bert/encoder',)
        body_train_op = super().get_train_op(loss,
                                             self.body_learning_rate,
                                             **kwargs)
        return body_train_op

    def save(self, exclude_scopes=('Optimizer',)) -> None:
        return super().save(exclude_scopes=exclude_scopes)

    def load(self,
             exclude_scopes=('Optimizer',
                             'learning_rate',
                             'momentum'),
             **kwargs) -> None:
        return super().load(exclude_scopes=exclude_scopes, **kwargs)

    def _decode_crf(self, feed_dict: Dict[tf.Tensor, np.ndarray]) -> List[np.ndarray]:
        logits, trans_params, mask, seq_lengths = self.sess.run([self.logits,
                                                                 self._transition_params,
                                                                 self.y_masks_ph,
                                                                 self.seq_lengths],
                                                                feed_dict=feed_dict)
        # iterate over the sentences because no batching in viterbi_decode
        y_pred = []
        for logit, sequence_length in zip(logits, seq_lengths):
            logit = logit[:int(sequence_length)]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            y_pred += [viterbi_seq]
        return y_pred

    def train_on_batch(self, *args, **kwargs) -> None:
        for task in self.tasks.values():
            kw = {inp_name: kwargs[inp_name] for inp_name in task.input_names}
            # TODO: check if there is reason to use different bert body lr for different tasks in literature
            task.train_on_batch(**kw, bert_body_learning_rate=max(self.get_learning_rate(), self.min_learning_rate))

    def __call__(self, *args, launch_name=None, **kwargs):
        if launch_name is None:
            if self.inference_launch_names is None:
                launch_names = list(self.launches.keys())
            else:
                launch_names = self.inference_launch_names
        else:
            launch_names = [launch_name]
        results = []
        for launch_name in launch_names:
            fetches = []
            feed_dict = {}
            tasks = list(self.launches_tasks[launch_name].values())
            for task in tasks:
                kw = {inp_name: kwargs[inp_name] for inp_name in task.input_names}
                task_fetches, task_feed_dict = task.get_sess_run_args(**kw)
                fetches.append(task_fetches)
                feed_dict.update(task_feed_dict)
            sess_run_res = self.sess.run(fetches, feed_dict=feed_dict)
            for task, srs in zip(tasks, sess_run_res):
                results += task.post_process_preds(srs)
        return results


@register("mt_bert_seq_tagging_task")
class MTBertSequenceTaggingTask:
    def __init__(
            self,
            task_name: str = "seq_tagging",
            n_tags: int = None,
            use_crf: bool = None,
            keep_prob: float = 1.,
            attention_probs_keep_probs: float = None,
            return_probas: bool = None,
            encoder_layer_ids: List[int] = None,
            learning_rate: float = 1e-3,
            in_names: List[str] = None,
            in_y_names: List[str] = None,
    ):
        self.task_name = task_name
        self.n_tags = n_tags
        self.use_crf = use_crf
        self.keep_prob = keep_prob
        self.return_probas = return_probas
        self.encoder_layer_ids = encoder_layer_ids
        self.init_head_learning_rate = learning_rate
        self.min_body_learning_rate = None
        self.head_learning_rate_multiplier = None
        self.in_names = in_names
        self.in_y_names = in_y_names

        self.bert = None
        self.shared_params = None
        self.shared_placeholders = None
        self.shared_feed_dict = None
        self.sess = None

    def build(self, bert_body, shared_params, shared_placeholders, shared_feed_dict, sess):
        self.bert = bert_body
        self.shared_params = shared_params
        self.head_learning_rate_multiplier = \
            self.init_head_learning_rate / self.shared_params['bert_body_learning_rate']
        mblr = self.shared_params.get('min_body_learning_rate')
        self.min_body_learning_rate = 0. if mblr is None else mblr
        self.shared_placeholders = shared_placeholders
        self.shared_feed_dict = shared_feed_dict
        self.sess = sess
        self._init_graph()
        self._init_optimizer()

    def _init_placeholders(self) -> None:
        self.y_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_ph')
        self.y_masks_ph = tf.placeholder(shape=(None, None),
                                         dtype=tf.int32,
                                         name='y_mask_ph')

    def _init_graph(self) -> None:
        self._init_placeholders()
        self.seq_lengths = tf.reduce_sum(self.y_masks_ph, axis=1)

        with tf.variable_scope(self.task_name):
            layer_weights = tf.get_variable('layer_weights_',
                                            shape=len(self.encoder_layer_ids),
                                            initializer=tf.ones_initializer(),
                                            trainable=True)
            layer_mask = tf.ones_like(layer_weights)
            layer_mask = tf.nn.dropout(layer_mask, self.shared_placeholders['encoder_keep_prob'])
            layer_weights *= layer_mask
            # to prevent zero division
            mask_sum = tf.maximum(tf.reduce_sum(layer_mask), 1.0)
            layer_weights = tf.unstack(layer_weights / mask_sum)
            # TODO: may be stack and reduce_sum is faster
            units = sum(w * l for w, l in zip(layer_weights, self.encoder_layers()))
            units = tf.nn.dropout(units, keep_prob=self.shared_placeholders['keep_prob'])
            if self.use_birnn:
                units, _ = bi_rnn(units,
                                  self.birnn_hidden_size,
                                  cell_type=self.birnn_cell_type,
                                  seq_lengths=self.seq_lengths,
                                  name='birnn')
                units = tf.concat(units, -1)
            # TODO: maybe add one more layer?
            logits = tf.layers.dense(units, units=self.n_tags, name="output_dense")

            self.logits = token_from_subtoken(logits, self.y_masks_ph)

            # CRF
            if self.use_crf:
                transition_params = tf.get_variable('Transition_Params',
                                                    shape=[self.n_tags, self.n_tags],
                                                    initializer=tf.zeros_initializer())
                log_likelihood, transition_params = \
                    tf.contrib.crf.crf_log_likelihood(self.logits,
                                                      self.y_ph,
                                                      self.seq_lengths,
                                                      transition_params)
                loss_tensor = -log_likelihood
                self._transition_params = transition_params

            self.y_predictions = tf.argmax(self.logits, -1)
            self.y_probas = tf.nn.softmax(self.logits, axis=2)

        with tf.variable_scope("loss"):
            tag_mask = self._get_tag_mask()
            y_mask = tf.cast(tag_mask, tf.float32)
            if self.use_crf:
                self.loss = tf.reduce_mean(loss_tensor)
            else:
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y_ph,
                                                                   logits=self.logits,
                                                                   weights=y_mask)

    def get_train_op(self, loss: tf.Tensor, bert_body_learning_rate: Union[tf.Tensor, float], **kwargs) -> tf.Operation:
        assert "learnable_scopes" not in kwargs, "learnable scopes unsupported"
        # train_op for bert variables
        kwargs['learnable_scopes'] = ('bert/encoder', 'bert/embeddings')
        if self.freeze_embeddings:
            kwargs['learnable_scopes'] = ('bert/encoder',)
        learning_rate = bert_body_learning_rate * self.head_learning_rate_multiplier
        bert_train_op = super().get_train_op(loss,
                                             bert_body_learning_rate,
                                             **kwargs)
        # train_op for ner head variables
        kwargs['learnable_scopes'] = (self.task_name,)
        head_train_op = super().get_train_op(loss,
                                             learning_rate,
                                             **kwargs)
        return tf.group(bert_train_op, head_train_op)

    def _init_optimizer(self) -> None:
        with tf.variable_scope('Optimizer'):
            self.global_step = tf.get_variable('global_step',
                                               shape=[],
                                               dtype=tf.int32,
                                               initializer=tf.constant_initializer(0),
                                               trainable=False)  # TODO: check this global step is not used in other subtasks
            # default optimizer for Bert is Adam with fixed L2 regularization

        if self.optimizer is None:
            self.train_op = \
                self.get_train_op(self.loss,
                                  learning_rate=self.shared_ph['learning_rate'],
                                  optimizer=AdamWeightDecayOptimizer,
                                  weight_decay_rate=self.weight_decay_rate,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-6,
                                  optimizer_scope_name='Optimizer',
                                  exclude_from_weight_decay=["LayerNorm",
                                                             "layer_norm",
                                                             "bias"])
        else:
            self.train_op = self.get_train_op(self.loss,
                                              learning_rate=self.shared_ph['learning_rate'],
                                              optimizer_scope_name='Optimizer')

        if self.optimizer is None:
            with tf.variable_scope('Optimizer'):
                new_global_step = self.global_step + 1
                self.train_op = tf.group(self.train_op, [self.global_step.assign(new_global_step)])

    def _build_basic_feed_dict(self, input_ids: tf.Tensor, input_masks: tf.Tensor,
                               token_types: Optional[tf.Tensor]=None, train: bool=False) -> dict:
        """Fills the feed_dict with the tensors defined in the basic class.
        You need to update this dict by the values of output placeholders
        and class-specific network inputs in your derived class.
        """
        feed_dict = {
            self.input_ids_ph: input_ids,
            self.input_masks_ph: input_masks,
        }
        if token_types is not None:
            feed_dict[self.token_types_ph] = token_types
        if train:
            feed_dict.update({
                self.learning_rate_ph: max(self.get_learning_rate(), self.min_learning_rate),
                self.keep_prob_ph: self.keep_prob,
                self.encoder_keep_prob_ph: 1.0 - self.encoder_dropout,
                self.is_train_ph: True,
            })

        return feed_dict

    def _build_feed_dict(self, input_ids, input_masks, y_masks, token_types, y=None, bert_body_learning_rate=None):
        sph = self.shared_placeholders
        train = y is not None
        feed_dict = {
            sph['input_ids']: input_ids,
            sph['input_masks']: input_masks,
            sph['token_types']: token_types,
        }
        if train:
            feed_dict.update({
                sph['learning_rate']: max(bert_body_learning_rate, self.min_body_learning_rate),
                sph['keep_prob']: self.keep_prob,
                sph['encoder_keep_prob']: 1.0 - self.encoder_dropout,
                sph['is_train']: True,
                self.y_ph: y,
            })
        feed_dict[self.y_masks_ph] = y_masks
        return feed_dict

    def _decode_crf(self, feed_dict: Dict[tf.Tensor, np.ndarray]) -> List[np.ndarray]:
        logits, trans_params, mask, seq_lengths = self.sess.run([self.logits,
                                                                 self._transition_params,
                                                                 self.y_masks_ph,
                                                                 self.seq_lengths],
                                                                feed_dict=feed_dict)
        # iterate over the sentences because no batching in viterbi_decode
        y_pred = []
        for logit, sequence_length in zip(logits, seq_lengths):
            logit = logit[:int(sequence_length)]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            y_pred += [viterbi_seq]
        return y_pred

    def train_on_batch(self,
                       input_ids: Union[List[List[int]], np.ndarray],
                       input_masks: Union[List[List[int]], np.ndarray],
                       y_masks: list,
                       y: list,
                       bert_body_learning_rate: float) -> Dict[str, float]:
        """

        Args:
            input_ids: batch of indices of subwords
            input_masks: batch of masks which determine what should be attended
            args: arguments passed  to _build_feed_dict
                and corresponding to additional input
                and output tensors of the derived class.
            kwargs: keyword arguments passed to _build_feed_dict
                and corresponding to additional input
                and output tensors of the derived class.

        Returns:
            dict with fields 'loss', 'head_learning_rate', and 'bert_learning_rate'
        """
        feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks, y)

        _, loss, lr = self.sess.run([self.train_op, self.loss, self.learning_rate_ph],
                                     feed_dict=feed_dict)
        return {'loss': loss,
                f'{self.task_name}_head_learning_rate': float(lr) * self.bert_learning_rate_multiplier,
                'bert_learning_rate': float(lr)}

    def get_sess_run_args(
            self,
            input_ids: Union[List[List[int]], np.ndarray],
            input_masks: Union[List[List[int]], np.ndarray],
            y_masks: Union[List[List[int]], np.ndarray],
            bert_features_qr: List[InputFeatures]):
        input_type_ids = [f.input_type_ids for f in bert_features_qr]
        feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks, input_type_ids)
        if self.return_probas:
            fetches = self.y_probas
        else:
            if self.use_crf:
                fetches = [self.logits, self._transition_params, self.y_masks_ph, self.seq_lengths]
            else:
                fetches = [self.y_predictions, self.seq_lengths]
        return fetches, feed_dict

    def post_process_preds(self, sess_run_res):
        if self.return_probas:
            pred = sess_run_res
        else:
            if self.use_crf:
                logits, trans_params, mask, seq_lengths = sess_run_res
                pred = []
                for logit, sequence_length in zip(logits, seq_lengths):
                    logit = logit[:int(sequence_length)]  # keep only the valid steps
                    viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                    pred += [viterbi_seq]
            else:
                pred, seq_lengths = sess_run_res
                pred = [p[:l] for l, p in zip(seq_lengths, pred)]
        return pred

    # def __call__(self,
    #              input_ids: Union[List[List[int]], np.ndarray],
    #              input_masks: Union[List[List[int]], np.ndarray],
    #              y_masks: Union[List[List[int]], np.ndarray],
    #              bert_features_qr: List[InputFeatures]) -> Union[List[List[int]], List[np.ndarray]]:
    #     input_type_ids = [f.input_type_ids for f in bert_features_qr]
    #     feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks, input_type_ids)
    #
    #     if self.return_probas:
    #         pred = self.sess.run(self.y_probas, feed_dict=feed_dict)
    #     else:
    #         if self.use_crf:
    #             pred = self._decode_crf(feed_dict)
    #         else:
    #             pred, seq_lengths = self.sess.run([self.y_predictions, self.seq_lengths], feed_dict=feed_dict)
    #             pred = [p[:l] for l, p in zip(seq_lengths, pred)]
    #     return pred

    def __call__(self,
                 input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray],
                 y_masks: Union[List[List[int]], np.ndarray],
                 bert_features_qr: List[InputFeatures]) -> Union[List[List[int]], List[np.ndarray]]:
        fetches, feed_dict = self.get_sess_run_args(input_ids, input_masks, y_masks, bert_features_qr)
        sess_run_res = self.sess.run(fetches, feed_dict=feed_dict)
        return self.post_process_preds(sess_run_res)


@register("mt_bert_classification_task")
class MTBertClassificationTask:
    def __init__(
            self,
            task_name: str = "classification",
            n_classes: int = None,
            return_probas: bool = None,
            one_hot_labels: bool = None,
            keep_prob: float = 1.,
            multilabel: bool = False,
            learning_rate: float = 2e-5,
            optimizer: str = "Adam",
            in_names: List[str] = None,
            in_y_names: List[str] = None,
    ):
        self.task_name = task_name
        self.n_classes = n_classes
        self.return_probas = return_probas
        self.one_hot_labels = one_hot_labels
        self.keep_prob = keep_prob
        self.multilabel = multilabel
        self.init_head_learning_rate = learning_rate
        self.head_learning_rate_multiplier = None
        self.min_body_learning_rate = None
        self.optimizer = optimizer
        self.in_names = in_names
        self.in_y_names = in_y_names

        self.bert = None
        self.shared_params = None
        self.shared_placeholders = None
        self.shared_feed_dict = None
        self.sess = None

        if self.multilabel and not self.one_hot_labels:
            raise RuntimeError('Use one-hot encoded labels for multilabel classification!')

        if self.multilabel and not self.return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

    def build(self, bert_body, shared_params, shared_placeholders, shared_feed_dict, sess):
        self.bert = bert_body
        self.shared_params = shared_params
        self.head_learning_rate_multiplier = \
            self.init_head_learning_rate / self.shared_params['bert_body_learning_rate']
        mblr = self.shared_params.get('min_body_learning_rate')
        self.min_body_learning_rate = 0. if mblr is None else mblr
        self.shared_placeholders = shared_placeholders
        self.shared_feed_dict = shared_feed_dict
        self.sess = sess
        self._init_graph()
        self._init_optimizer()

    def _init_placeholders(self):
        if not self.one_hot_labels:
            self.y_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='y_ph')
        else:
            self.y_ph = tf.placeholder(shape=(None, self.n_classes), dtype=tf.float32, name='y_ph')

    def _init_graph(self):
        self._init_placeholders()

        output_layer = self.bert.get_pooled_output()
        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [self.n_classes, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [self.n_classes], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            output_layer = tf.nn.dropout(output_layer, keep_prob=self.keep_prob_ph)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            if self.one_hot_labels:
                one_hot_labels = self.y_ph
            else:
                one_hot_labels = tf.one_hot(self.y_ph, depth=self.n_classes, dtype=tf.float32)

            self.y_predictions = tf.argmax(logits, axis=-1)
            if not self.multilabel:
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                self.y_probas = tf.nn.softmax(logits, axis=-1)
                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                self.loss = tf.reduce_mean(per_example_loss)
            else:
                self.y_probas = tf.nn.sigmoid(logits)
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))

    def get_train_op(self, loss: tf.Tensor, bert_body_learning_rate: Union[tf.Tensor, float], **kwargs) -> tf.Operation:
        assert "learnable_scopes" not in kwargs, "learnable scopes unsupported"
        # train_op for bert variables
        kwargs['learnable_scopes'] = ('bert/encoder', 'bert/embeddings')
        if self.freeze_embeddings:
            kwargs['learnable_scopes'] = ('bert/encoder',)
        learning_rate = bert_body_learning_rate * self.head_learning_rate_multiplier
        bert_train_op = super().get_train_op(loss,
                                             bert_body_learning_rate,
                                             **kwargs)
        # train_op for ner head variables
        kwargs['learnable_scopes'] = (self.task_name,)
        head_train_op = super().get_train_op(loss,
                                             learning_rate,
                                             **kwargs)
        return tf.group(bert_train_op, head_train_op)

    def _init_optimizer(self):
        with tf.variable_scope('Optimizer'):
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            # default optimizer for Bert is Adam with fixed L2 regularization
            if self.optimizer is None:

                self.train_op = self.get_train_op(self.loss, bert_body_learning_rate=self.learning_rate_ph,
                                                  optimizer=AdamWeightDecayOptimizer,
                                                  weight_decay_rate=self.weight_decay_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=1e-6,
                                                  exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
                                                  )
            else:
                self.train_op = self.get_train_op(self.loss, bert_body_learning_rate=self.learning_rate_ph)

            if self.optimizer is None:
                new_global_step = self.global_step + 1
                self.train_op = tf.group(self.train_op, [self.global_step.assign(new_global_step)])

    def _build_feed_dict(self, input_ids, input_masks, token_types, y=None, bert_body_learning_rate=None):
        sph = self.shared_placeholders
        feed_dict = {
            sph['input_ids_ph']: input_ids,
            sph['input_masks']: input_masks,
            sph['token_types_ph']: token_types,
        }
        if y is not None:
            feed_dict.update({
                self.y_ph: y,
                sph['learning_rate']: max(bert_body_learning_rate, self.min_learning_rate),
                sph['keep_prob']: self.keep_prob,
                sph['is_train']: True,
            })

        return feed_dict

    def train_on_batch(
            self,
            features: List[InputFeatures],
            y: Union[List[int], List[List[int]]],
            bert_body_learning_rate: float,
    ) -> Dict:
        """Train model on given batch.
        This method calls train_op using features and y (labels).

        Args:
            features: batch of InputFeatures
            y: batch of labels (class id or one-hot encoding)

        Returns:
            dict with loss and learning_rate values

        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]

        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids, y, bert_body_learning_rate)

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return {'loss': loss, f'{self.task_name}_learning_rate': feed_dict[self.learning_rate_ph]}

    def get_sess_run_args(self, features: List[InputFeatures]):
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]
        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids)
        fetches = self.y_probas if self.return_probas else self.y_predictions
        return fetches, feed_dict

    def post_process_preds(self, sess_run_res):
        return sess_run_res

    def __call__(
            self,
            features: List[InputFeatures]) -> Union[List[int], List[List[float]]]:
        """Make prediction for given features (texts).

        Args:
            features: batch of InputFeatures

        Returns:
            predicted classes or probabilities of each class

        """
        fetches, feed_dict = self.get_sess_run_args(features)
        return self.sess.run(fetches, feed_dict=feed_dict)


@register("mt_bert_reuser")
class MTBertReUser:
    def __init__(self, mt_bert, launch_name):
        self.mt_bert = mt_bert
        self.launch_name = launch_name

    def __call__(self, *args, **kwargs):
        return self.mt_bert(*args, **kwargs, launch_name=self.launch_name)
