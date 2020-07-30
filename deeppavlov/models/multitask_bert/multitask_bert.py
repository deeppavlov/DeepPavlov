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

import copy
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer
from bert_dp.preprocessing import InputFeatures
from overrides import overrides

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.layers.tf_layers import bi_rnn
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.models.bert.bert_sequence_tagger import token_from_subtoken

log = getLogger(__name__)


class MTBertTask(ABC):
    """Abstract class for multitask BERT tasks. Objects of its subclasses are linked with BERT body when
    ``MultiTaskBert.build`` method is called. Training is performed with ``MultiTaskBert.train_on_batch`` method is
    called. The objects of classes derived from ``MTBertTask`` don't have ``__call__`` method. Instead they have
    ``get_sess_run_infer_args`` and ``post_process_preds`` methods, which are called from ``call`` method of
    ``MultiTaskBert`` class. ``get_sess_run_infer_args`` method returns fetches and feed_dict for inference and
    ``post_process_preds`` method retrieves predictions from computed fetches. Classes derived from ``MTBertTask``
    must ``get_sess_run_train_args`` method that returns fetches and feed_dict for training.

    Args:
        keep_prob: dropout keep_prob for non-BERT layers
        return_probas: set this to ``True`` if you need the probabilities instead of raw answers
        learning_rate: learning rate of BERT head
    """

    def __init__(
            self,
            keep_prob: float = 1.,
            return_probas: bool = None,
            learning_rate: float = 1e-3,
    ):
        self.keep_prob = keep_prob
        self.return_probas = return_probas
        self.init_head_learning_rate = learning_rate
        self.min_body_learning_rate = None
        self.head_learning_rate_multiplier = None

        self.bert = None
        self.optimizer_params = None
        self.shared_ph = None
        self.shared_feed_dict = None
        self.sess = None
        self.get_train_op_func = None
        self.freeze_embeddings = None
        self.bert_head_variable_scope = None

    def build(
            self,
            bert_body: BertModel,
            optimizer_params: Dict[str, Union[str, float]],
            shared_placeholders: Dict[str, tf.placeholder],
            sess: tf.Session,
            mode: str,
            get_train_op_func: Callable,
            freeze_embeddings: bool,
            bert_head_variable_scope: str) -> None:
        """Initiates building of the BERT head and initializes optimizer parameters, placeholders that are common for
        all tasks.

        Args:
            bert_body: instance of ``BertModel``.
            optimizer_params: a dictionary with four fields:
                ``'optimizer'`` (``str``) -- a name of optimizer class,
                ``'body_learning_rate'`` (``float``) -- initial value of BERT body learning rate,
                ``'min_body_learning_rate'`` (``float``) -- min BERT body learning rate for learning rate decay,
                ``'weight_decay_rate'`` (``float``) -- L2 weight decay for ``AdamWeightDecayOptimizer``
            shared_placeholders: a dictionary with placeholders used in all tasks. The dictionary contains fields
                 ``'input_ids'``, ``'input_masks'``, ``'learning_rate'``, ``'keep_prob'``, ``'is_train'``,
                 ``'token_types'``.
            sess: current ``tf.Session`` instance
            mode: ``'train'`` or ``'inference'``
            get_train_op_func: a function returning ``tf.Operation`` and with signature similar to
                ``LRScheduledTFModel.get_train_op`` without ``self`` argument. It is a function returning train
                operation for specified loss and variable scopes.
            freeze_embeddings: set ``False`` to train input embeddings.
            bert_head_variable_scope: variable scope for BERT head.
        """
        self.bert_head_variable_scope = bert_head_variable_scope
        self.get_train_op_func = get_train_op_func
        self.freeze_embeddings = freeze_embeddings
        self.bert = bert_body
        self.optimizer_params = optimizer_params
        if mode == 'train':
            self.head_learning_rate_multiplier = \
                self.init_head_learning_rate / self.optimizer_params['body_learning_rate']
        else:
            self.head_learning_rate_multiplier = 0
        mblr = self.optimizer_params.get('min_body_learning_rate')
        self.min_body_learning_rate = 0. if mblr is None else mblr
        self.shared_ph = shared_placeholders
        self.sess = sess
        self._init_graph()
        if mode == 'train':
            self._init_optimizer()

    @abstractmethod
    def _init_graph(self) -> None:
        """Build BERT head, initialize task specific placeholders, create attributes containing output probabilities
        and model loss. Optimizer initialized not in this method but in ``_init_optimizer``."""
        pass

    def get_train_op(self, loss: tf.Tensor, body_learning_rate: Union[tf.Tensor, float], **kwargs) -> tf.Operation:
        """Return operation for the task training. Head learning rate is calculated as a product of
        ``body_learning_rate`` and quotient of initial head learning rate and initial body learning rate.

        Args:
            loss: the task loss
            body_learning_rate: the learning rate for the BERT body

        Returns:
            train operation for the task
        """
        assert "learnable_scopes" not in kwargs, "learnable scopes unsupported"
        # train_op for bert variables
        kwargs['learnable_scopes'] = ('bert/encoder', 'bert/embeddings')
        if self.freeze_embeddings:
            kwargs['learnable_scopes'] = ('bert/encoder',)
        learning_rate = body_learning_rate * self.head_learning_rate_multiplier
        bert_train_op = self.get_train_op_func(loss, body_learning_rate, **kwargs)
        # train_op for ner head variables
        kwargs['learnable_scopes'] = (self.bert_head_variable_scope,)
        head_train_op = self.get_train_op_func(loss, learning_rate, **kwargs)
        return tf.group(bert_train_op, head_train_op)

    def _init_optimizer(self) -> None:
        with tf.variable_scope(self.bert_head_variable_scope):
            with tf.variable_scope('Optimizer'):
                self.global_step = tf.get_variable('global_step',
                                                   shape=[],
                                                   dtype=tf.int32,
                                                   initializer=tf.constant_initializer(0),
                                                   trainable=False)
                # default optimizer for Bert is Adam with fixed L2 regularization

            if self.optimizer_params.get('optimizer') is None:
                self.train_op = \
                    self.get_train_op(
                        self.loss,
                        body_learning_rate=self.shared_ph['learning_rate'],
                        optimizer=AdamWeightDecayOptimizer,
                        weight_decay_rate=self.optimizer_params.get('weight_decay_rate', 1e-6),
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-6,
                        optimizer_scope_name='Optimizer',
                        exclude_from_weight_decay=["LayerNorm",
                                                   "layer_norm",
                                                   "bias"])
            else:
                self.train_op = self.get_train_op(self.loss,
                                                  body_learning_rate=self.shared_ph['learning_rate'],
                                                  optimizer_scope_name='Optimizer')

            if self.optimizer_params.get('optimizer') is None:
                with tf.variable_scope('Optimizer'):
                    new_global_step = self.global_step + 1
                    self.train_op = tf.group(self.train_op, [self.global_step.assign(new_global_step)])

    def _build_feed_dict(self, input_ids, input_masks, token_types, y=None, body_learning_rate=None):
        sph = self.shared_ph
        train = y is not None
        feed_dict = {
            sph['input_ids']: input_ids,
            sph['input_masks']: input_masks,
            sph['token_types']: token_types,
            sph['is_train']: train,
        }
        if train:
            feed_dict.update({
                sph['learning_rate']: body_learning_rate,
                self.y_ph: y,
                sph['keep_prob']: self.keep_prob,
            })
        return feed_dict

    def train_on_batch(self, *args, **kwargs) -> Dict[str, float]:
        """Trains the task on one batch. This method will work correctly if you override ``get_sess_run_train_args``
        for your task.

        Args:
            kwargs: the keys are ``body_learning_rate`` and ``"in"`` and ``"in_y"`` params for the task.

        Returns:
            dictionary with calcutated task loss and body and head learning rates.
        """
        fetches, feed_dict = self.get_sess_run_train_args(*args, **kwargs)
        _, loss = self.sess.run(fetches, feed_dict=feed_dict)
        return {f'{self.bert_head_variable_scope}_loss': loss,
                f'{self.bert_head_variable_scope}_head_learning_rate':
                    float(kwargs['body_learning_rate']) * self.head_learning_rate_multiplier,
                'bert_body_learning_rate': kwargs['body_learning_rate']}

    @abstractmethod
    def get_sess_run_infer_args(self, *args) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for inference. Fetches are lists of tensors and feed_dict is dictionary
        with placeholder values required for fetches computation. The method is used inside ``MultiTaskBert``
        ``__call__`` method.

        If ``self.return_probas`` is ``True`` fetches contains probabilities tensor and predictions tensor otherwise.

        Overriding methods take task inputs as positional arguments.

        ATTENTION! Let ``get_sess_run_infer_args`` method have ``n_x_args`` arguments. Then the order of first
        ``n_x_args`` arguments of ``get_sess_run_train_args`` method arguments has to match the order of
        ``get_sess_run_infer_args`` arguments.

        Args:
            args: task inputs.

        Returns:
            fetches and feed_dict
        """
        pass

    @abstractmethod
    def get_sess_run_train_args(self, *args) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for task ``train_on_batch`` method.

        Overriding methods take task inputs as positional arguments.

        ATTENTION! Let ``get_sess_run_infer_args`` method have ``n_x_args`` arguments. Then the order of first
        ``n_x_args`` arguments of ``get_sess_run_train_args`` method arguments has to match the order of
        ``get_sess_run_infer_args`` arguments.

        Args:
            args: task inputs followed by expect outputs.

        Returns:
            fetches and feed_dict
        """
        pass

    @abstractmethod
    def post_process_preds(self, sess_run_res: list) -> list:
        """Post process results of ``tf.Session.run`` called for task inference. Called from method
        ``MultiTaskBert.__call__``.

        Args:
            sess_run_res: computed fetches from ``get_sess_run_infer_args`` method

        Returns:
            post processed results
        """
        pass


@register("mt_bert_seq_tagging_task")
class MTBertSequenceTaggingTask(MTBertTask):
    """BERT head for text tagging. It predicts a label for every token (not subtoken) in the text.
    You can use it for sequence labelling tasks, such as morphological tagging or named entity recognition.
    Objects of this class should be passed to the constructor of ``MultiTaskBert`` class in param ``tasks``.

    Args:
        n_tags: number of distinct tags
        use_crf: whether to use CRF on top or not
        use_birnn: whether to use bidirection rnn after BERT layers.
            For NER and morphological tagging we usually set it to ``False`` as otherwise the model overfits
        birnn_cell_type: the type of Bidirectional RNN. Either ``"lstm"`` or ``"gru"``
        birnn_hidden_size: number of hidden units in the BiRNN layer in each direction
        keep_prob: dropout keep_prob for non-Bert layers
        encoder_dropout: dropout probability of encoder output layer
        return_probas: set this to ``True`` if you need the probabilities instead of raw answers
        encoder_layer_ids: list of averaged layers from Bert encoder (layer ids)
            optimizer: name of ``tf.train.*`` optimizer or None for ``AdamWeightDecayOptimizer``
            weight_decay_rate: L2 weight decay for ``AdamWeightDecayOptimizer``
        learning_rate: learning rate of BERT head
    """

    def __init__(
            self,
            n_tags: int = None,
            use_crf: bool = None,
            use_birnn: bool = False,
            birnn_cell_type: str = 'lstm',
            birnn_hidden_size: int = 128,
            keep_prob: float = 1.,
            encoder_dropout: float = 0.,
            return_probas: bool = None,
            encoder_layer_ids: List[int] = None,
            learning_rate: float = 1e-3,
    ):
        super().__init__(keep_prob, return_probas, learning_rate)
        self.n_tags = n_tags
        self.use_crf = use_crf
        self.use_birnn = use_birnn
        self.birnn_cell_type = birnn_cell_type
        self.birnn_hidden_size = birnn_hidden_size
        self.encoder_dropout = encoder_dropout
        self.encoder_layer_ids = encoder_layer_ids

    def _init_placeholders(self) -> None:
        self.y_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_ph')
        self.y_masks_ph = tf.placeholder(shape=(None, None),
                                         dtype=tf.int32,
                                         name='y_mask_ph')
        self.encoder_keep_prob = tf.placeholder_with_default(1.0, shape=[], name='encoder_keep_prob_ph')

    def _init_graph(self) -> None:
        with tf.variable_scope(self.bert_head_variable_scope):
            self._init_placeholders()
            self.seq_lengths = tf.reduce_sum(self.y_masks_ph, axis=1)

            layer_weights = tf.get_variable('layer_weights_',
                                            shape=len(self.encoder_layer_ids),
                                            initializer=tf.ones_initializer(),
                                            trainable=True)
            layer_mask = tf.ones_like(layer_weights)
            layer_mask = tf.nn.dropout(layer_mask, self.encoder_keep_prob)
            layer_weights *= layer_mask
            # to prevent zero division
            mask_sum = tf.maximum(tf.reduce_sum(layer_mask), 1.0)
            layer_weights = tf.unstack(layer_weights / mask_sum)
            # TODO: may be stack and reduce_sum is faster
            units = sum(w * l for w, l in zip(layer_weights, self.encoder_layers()))
            units = tf.nn.dropout(units, keep_prob=self.shared_ph['keep_prob'])
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

    def _get_tag_mask(self) -> tf.Tensor:
        """
        Returns: tag_mask,
            a mask that selects positions corresponding to word tokens (not padding and ``CLS``)
        """
        max_length = tf.reduce_max(self.seq_lengths)
        one_hot_max_len = tf.one_hot(self.seq_lengths - 1, max_length)
        tag_mask = tf.cumsum(one_hot_max_len[:, ::-1], axis=1)[:, ::-1]
        return tag_mask

    def encoder_layers(self):
        """
        Returns: the output of BERT layers specified in ``self.encoder_layers_ids``
        """
        return [self.bert.all_encoder_layers[i] for i in self.encoder_layer_ids]

    @overrides
    def _build_feed_dict(self, input_ids, input_masks, y_masks, y=None, body_learning_rate=None):
        token_types = np.zeros(np.array(input_ids).shape)
        sph = self.shared_ph
        train = y is not None
        feed_dict = super()._build_feed_dict(input_ids, input_masks, token_types, y, body_learning_rate)
        if train:
            feed_dict[self.encoder_keep_prob] = 1.0 - self.encoder_dropout
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

    def get_sess_run_infer_args(
            self,
            input_ids: Union[List[List[int]], np.ndarray],
            input_masks: Union[List[List[int]], np.ndarray],
            y_masks: Union[List[List[int]], np.ndarray],
    ) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for model inference. The method is called from ``MultiTaskBert.__call__``.

        Args:
            input_ids: indices of the subwords in vocabulary
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word

        Returns:
            list of fetches and feed_dict
        """
        feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks)
        if self.return_probas:
            fetches = self.y_probas
        else:
            if self.use_crf:
                fetches = [self.logits, self._transition_params, self.y_masks_ph, self.seq_lengths]
            else:
                fetches = [self.y_predictions, self.seq_lengths]
        return fetches, feed_dict

    def get_sess_run_train_args(
            self,
            input_ids: Union[List[List[int]], np.ndarray],
            input_masks: Union[List[List[int]], np.ndarray],
            y_masks: Union[List[List[int]], np.ndarray],
            y: Union[List[List[int]], np.ndarray],
            body_learning_rate: float) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for model ``train_on_batch`` method.

        Args:
            input_ids: indices of the subwords in vocabulary
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word
            y: indices of ground truth tags
            body_learning_rate: learning rate for BERT body

        Returns:
            list of fetches and feed_dict
        """
        feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks, y=y, body_learning_rate=body_learning_rate)
        fetches = [self.train_op, self.loss]
        return fetches, feed_dict

    def post_process_preds(self, sess_run_res: List[np.ndarray]) -> Union[List[List[int]], List[np.ndarray]]:
        """Decodes CRF if needed and returns predictions or probabilities.

        Args:
            sess_run_res: list of computed fetches gathered by ``get_sess_run_infer_args``

        Returns:
            predictions or probabilities depending on ``return_probas`` attribute
        """
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


@register("mt_bert_classification_task")
class MTBertClassificationTask(MTBertTask):
    """Task for text classification.

    It uses output from [CLS] token and predicts labels using linear transformation.

    Args:
        n_classes: number of classes
        return_probas: set ``True`` if return class probabilities instead of most probable label needed
        one_hot_labels: set ``True`` if one-hot encoding for labels is used
        keep_prob: dropout keep_prob for non-BERT layers
        multilabel: set ``True`` if it is multi-label classification
        learning_rate: learning rate of BERT head
        optimizer: name of ``tf.train.*`` optimizer or ``None`` for ``AdamWeightDecayOptimizer``
    """

    def __init__(
            self,
            n_classes: int = None,
            return_probas: bool = None,
            one_hot_labels: bool = None,
            keep_prob: float = 1.,
            multilabel: bool = False,
            learning_rate: float = 2e-5,
            optimizer: str = "Adam",
    ):
        super().__init__(keep_prob, return_probas, learning_rate)
        self.n_classes = n_classes
        self.one_hot_labels = one_hot_labels
        self.multilabel = multilabel

        if self.multilabel and not self.one_hot_labels:
            raise RuntimeError('Use one-hot encoded labels for multilabel classification!')

        if self.multilabel and not self.return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

    def _init_placeholders(self):
        if not self.one_hot_labels:
            self.y_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='y_ph')
        else:
            self.y_ph = tf.placeholder(shape=(None, self.n_classes), dtype=tf.float32, name='y_ph')

    def _init_graph(self):
        with tf.variable_scope(self.bert_head_variable_scope):
            self._init_placeholders()

            output_layer = self.bert.get_pooled_output()
            hidden_size = output_layer.shape[-1].value

            output_weights = tf.get_variable(
                "output_weights", [self.n_classes, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [self.n_classes], initializer=tf.zeros_initializer())

            with tf.variable_scope("loss"):
                output_layer = tf.nn.dropout(output_layer, keep_prob=self.shared_ph['keep_prob'])
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

    def get_sess_run_train_args(
            self,
            features: List[InputFeatures],
            y: Union[List[int], List[List[int]]],
            body_learning_rate: float) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for model ``train_on_batch`` method.

        Args:
            features: text features created by BERT preprocessor.
            y: batch of labels (class id or one-hot encoding)
            body_learning_rate: learning rate for BERT body

        Returns:
            list of fetches and feed_dict
        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]
        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids, y=y,
                                          body_learning_rate=body_learning_rate)
        fetches = [self.train_op, self.loss]
        return fetches, feed_dict

    def get_sess_run_infer_args(
            self,
            features: List[InputFeatures]) -> Tuple[List[tf.Tensor], Dict[tf.placeholder, Any]]:
        """Returns fetches and feed_dict for model inference. The method is called from ``MultiTaskBert.__call__``.

        Args:
            features: text features created by BERT preprocessor.

        Returns:
            list of fetches and feed_dict
        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.input_mask for f in features]
        input_type_ids = [f.input_type_ids for f in features]
        feed_dict = self._build_feed_dict(input_ids, input_masks, input_type_ids)
        fetches = self.y_probas if self.return_probas else self.y_predictions
        return fetches, feed_dict

    def post_process_preds(self, sess_run_res):
        """Returns ``tf.Session.run`` results for inference without changes."""
        return sess_run_res


@register('mt_bert')
class MultiTaskBert(LRScheduledTFModel):
    """The component for multi-task BERT. It builds the BERT body, launches building of BERT heads. 

    The component aggregates components implementing BERT heads. The head components are called tasks.
    ``__call__`` and ``train_on_batch`` methods of ``MultiTaskBert`` are used for inference and training of 
    BERT heads. BERT head components, which are derived from ``MTBertTask``, can be used only inside this class.

    One training iteration consists of one ``train_on_batch`` call for every task.
    
    If ``inference_task_names`` is not ``None``, then the component is created for training. Otherwise, the 
    component is created for inference. If component is created for inference, several tasks can be run 
    simultaneously. For explanation see parameter ``inference_task_names`` description.

    Args:
        tasks: a dictionary. Task names are dictionary keys and objects of ``MTBertTask`` subclasses are dictionary
            values. Task names are used as variable scopes in computational graph so it is important to use same
            names in multi-task BERT train and inference configuration files.
        bert_config_file: path to BERT configuration file
        pretrained_bert: pre-trained BERT checkpoint
        attention_probs_keep_prob: keep_prob for BERT self-attention layers
        hidden_keep_prob: keep_prob for BERT hidden layers
        body_learning_rate: learning rate of BERT body
        min_body_learning_rate: min value of body learning rate if learning rate decay is used
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after ``learning_rate_drop_patience`` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        clip_norm: clip gradients by norm
        freeze_embeddings: set to False to train input embeddings
        inference_task_names: names of tasks on which inference is done.
            If this parameter is provided, the component is created for inference, else the component is created for
            training. 

            If ``inference_task_names`` is a string, then it is a name of the task called separately from other tasks
            (in individual ``tf.Session.run`` call).

            If ``inference_task_names`` is a ``list``, then elements of this list are either strings or lists of
            strings. You can combine these options. For example, ``["task_name1", ["task_name2", "task_name3"],
            ["task_name4", "task_name5"]]``.

            If an element of ``inference_task_names`` list is a string, the element is a name of the task that is
            computed when ``__call__`` method is called.

            If an element of the ``inference_task_names`` parameter is a list of strings
            ``["task_name1", "task_name2", ...]``, then tasks ``"task_name1"``, ``"task_name2"`` and so on are run
            simultaneously in ``tf.Session.run`` call. This option is available if tasks ``"task_name1"``,
            ``"task_name2"`` and so on have common inputs. Despite the fact that tasks share inputs, if positional
            arguments are used in methods ``__call__`` and ``train_on_batch``, all arguments are passed individually.
            For instance, if ``"task_name1"``, ``"task_name2"``, and ``"task_name3"`` all take an argument with name
            ``x`` in the model pipe, then the ``__call__`` method takes arguments ``(x, x, x)``.
        in_distribution: The distribution of variables listed in the ``"in"`` config parameter between tasks. 
            ``in_distribution`` can be ``None`` if only 1 task is called. In that case all variables
            listed in ``"in"`` are arguments of 1 task. 

            ``in_distribution`` can be a dictionary of ``int``. If that is the case, then keys of ``in_distribution``
            are task names and values are numbers of variables from ``"in"`` parameter of config which are inputs of
            corresponding task. The variables in ``"in"`` parameter have to be in the same order the tasks are listed
            in ``in_distribution``.
            
            ``in_distribution`` can be a dictionary of lists of ``str``. Strings are names of variables from ``"in"``
            configuration parameter. If ``"in"`` parameter is a list, then ``in_distribution`` works the same way as
            when ``in_distribution`` is dictionary of ``int``. Values of ``in_distribution``, which are lists, are
            replaced by their lengths. If ``"in"`` parameter in component config is a dictionary, then the order of
            strings in ``in_distribution`` values has to match the order of arguments of ``train_on_batch`` and 
            ``get_sess_run_infer_args`` methods of task components.
        in_y_distribution: The same as ``in_distribution`` for ``"in_y"`` config parameter.
    """
    def __init__(self,
                 tasks: Dict[str, MTBertTask],
                 bert_config_file: str,
                 pretrained_bert: str = None,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 optimizer: str = None,
                 weight_decay_rate: float = 1e-6,
                 body_learning_rate: float = 1e-3,
                 min_body_learning_rate: float = 1e-7,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: float = 1.0,
                 freeze_embeddings: bool = True,
                 inference_task_names: Optional[Union[str, List[Union[List[str], str]]]] = None,
                 in_distribution: Optional[Dict[str, Union[int, List[str]]]] = None,
                 in_y_distribution: Optional[Dict[str, Union[int, List[str]]]] = None,
                 **kwargs) -> None:
        super().__init__(learning_rate=body_learning_rate,
                         learning_rate_drop_div=learning_rate_drop_div,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         load_before_drop=load_before_drop,
                         clip_norm=clip_norm,
                         **kwargs)
        self.optimizer_params = {
            "optimizer": optimizer,
            "body_learning_rate": body_learning_rate,
            "min_body_learning_rate": min_body_learning_rate,
            "weight_decay_rate": weight_decay_rate
        }
        self.freeze_embeddings = freeze_embeddings
        self.tasks = tasks

        if inference_task_names is not None and isinstance(inference_task_names, str):
            inference_task_names = [inference_task_names]
        self.inference_task_names = inference_task_names

        self.mode = 'train' if self.inference_task_names is None else 'inference'

        self.shared_ph = None

        self.bert_config = BertConfig.from_json_file(str(expand_path(bert_config_file)))

        if attention_probs_keep_prob is not None:
            self.bert_config.attention_probs_dropout_prob = 1.0 -attention_probs_keep_prob
        if hidden_keep_prob is not None:
            self.bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self._init_bert_body_graph()
        self.build_tasks()

        self.sess.run(tf.global_variables_initializer())

        if pretrained_bert is not None:
            pretrained_bert = str(expand_path(pretrained_bert))
            if tf.train.checkpoint_exists(pretrained_bert) \
                    and not (self.load_path and tf.train.checkpoint_exists(str(self.load_path.resolve()))) \
                    and self.mode == 'train':
                log.info('[initializing model with Bert from {}]'.format(pretrained_bert))
                var_list = self._get_saveable_variables(
                    exclude_scopes=('Optimizer', 'learning_rate', 'momentum') + tuple(self.tasks.keys()))
                saver = tf.train.Saver(var_list)
                saver.restore(self.sess, pretrained_bert)

        if self.load_path is not None:
            self.load()
        self.in_distribution = in_distribution
        self.in_y_distribution = in_y_distribution

    def build_tasks(self):
        def get_train_op(*args, **kwargs):
            return self.get_train_op(*args, **kwargs)
        for task_name, task_obj in self.tasks.items():
            task_obj.build(
                bert_body=self.bert,
                optimizer_params=self.optimizer_params,
                shared_placeholders=self.shared_ph,
                sess=self.sess,
                mode=self.mode,
                get_train_op_func=get_train_op,
                freeze_embeddings=self.freeze_embeddings,
                bert_head_variable_scope=task_name
            )

    def _init_shared_placeholders(self) -> None:
        self.shared_ph = {
            'input_ids': tf.placeholder(shape=(None, None),
                                        dtype=tf.int32,
                                        name='token_indices_ph'),
            'input_masks': tf.placeholder(shape=(None, None),
                                          dtype=tf.int32,
                                          name='token_mask_ph'),
            'learning_rate': tf.placeholder_with_default(0.0, shape=[], name='learning_rate_ph'),
            'keep_prob': tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph'),
            'is_train': tf.placeholder_with_default(False, shape=[], name='is_train_ph')}
        self.shared_ph['token_types'] = tf.placeholder_with_default(
                tf.zeros_like(self.shared_ph['input_ids'], dtype=tf.int32),
                shape=self.shared_ph['input_ids'].shape,
                name='token_types_ph')

    def _init_bert_body_graph(self) -> None:
        self._init_shared_placeholders()
        sph = self.shared_ph
        self.bert = BertModel(config=self.bert_config,
                              is_training=sph['is_train'],
                              input_ids=sph['input_ids'],
                              input_mask=sph['input_masks'],
                              token_type_ids=sph['token_types'],
                              use_one_hot_embeddings=False)

    def save(self, exclude_scopes=('Optimizer', 'learning_rate', 'momentum')) -> None:
        return super().save(exclude_scopes=exclude_scopes)

    def load(self,
             exclude_scopes=('Optimizer',
                             'learning_rate',
                             'momentum'),
             **kwargs) -> None:
        return super().load(exclude_scopes=exclude_scopes, **kwargs)

    def train_on_batch(self, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        """Calls ``train_on_batch`` methods for every task. This method takes ``args`` or ``kwargs`` but not both.
        The order of ``args`` is the same as the order of tasks in the component parameters:

        .. highlight:: python
        .. code-block:: python

           args = [
               task1_in_x[0],
               task1_in_x[1],
               task1_in_x[2],
               ...
               task1_in_y[0],
               task1_in_y[1],
               ...
               task2_in_x[0],
               ...
           ]
 
        If ``kwargs`` are used and ``in_distribution`` and ``in_y_distribution`` attributes are dictionaries of lists
        of strings, then keys of ``kwargs`` have to be same as strings in ``in_distribution`` and
        ``in_y_distribution``. If ``in_distribution`` and ``in_y_distribution`` are dictionaries of ``int``, then
        ``kwargs`` values are treated the same way as ``args``.

        Args:
            args: task inputs and expected outputs
            kwargs: task inputs and expected outputs

        Returns:
            dictionary of dictionaries with task losses and learning rates.
        """
        # TODO: test passing arguments as args
        if args and kwargs:
            raise ValueError("You can use either args or kwargs not both")
        n_in = sum([len(inp) if isinstance(inp, list) else inp for inp in self.in_distribution.values()])
        if args:
            args_in, args_in_y = args[:n_in], args[n_in:]
            in_by_tasks = self._distribute_arguments_by_tasks(args_in, {}, list(self.tasks.keys()), "in")
            in_y_by_tasks = self._distribute_arguments_by_tasks(args_in_y, {}, list(self.tasks.keys()), "in_y")
        else:
            kwargs_in, kwargs_in_y = {}, {}
            for i, (k, v) in enumerate(kwargs.items()):
                if i < n_in:
                    kwargs_in[k] = v
                else:
                    kwargs_in_y[k] = v
            in_by_tasks = self._distribute_arguments_by_tasks({}, kwargs_in, list(self.tasks.keys()), "in")
            in_y_by_tasks = self._distribute_arguments_by_tasks({}, kwargs_in_y, list(self.tasks.keys()), "in_y")
        train_on_batch_results = {}
        for task_name, task in self.tasks.items():
            train_on_batch_results.update(
                task.train_on_batch(
                    *in_by_tasks[task_name],
                    *in_y_by_tasks[task_name], 
                    body_learning_rate=max(self.get_learning_rate(), self.optimizer_params['min_body_learning_rate'])
                )
            )
        for k, v in train_on_batch_results.items():
            train_on_batch_results[k] = float(f"{v:.3}")
        return train_on_batch_results

    @staticmethod
    def _unite_task_feed_dicts(d1, d2, task_name):
        d = copy.copy(d1)
        for k, v in d2.items():
            if k in d:
                comp = v != d[k]
                if isinstance(comp, np.ndarray):
                    comp = comp.any()
                if comp:
                    raise ValueError(
                        f"Value of placeholder '{k}' for task '{task_name}' does not match value of this placeholder "
                        "in other tasks")
            else:
                d[k] = v
        return d

    def _distribute_arguments_by_tasks(self, args, kwargs, task_names, what_to_distribute, in_distribution=None):
        if args and kwargs:
            raise ValueError("You may use args or kwargs but not both")

        if what_to_distribute == "in":
            if in_distribution is not None:
                distribution = in_distribution
            else:
                distribution = self.in_distribution
        elif what_to_distribute == "in_y":
            if in_distribution is not None:
                raise ValueError(
                    f"If parameter `what_to_distribute` is 'in_y', parameter `in_distribution` has to be `None`. "
                    f"in_distribution = {in_distribution}")
            distribution = self.in_y_distribution
        else:
            raise ValueError(f"`what_to_distribute` can be 'in' or 'in_y', {repr(what_to_distribute)} is given")

        if distribution is None:
            if len(task_names) != 1:
                raise ValueError(f"If no `{what_to_distribute}_distribution` is not provided there have to be only 1"
                                 "task for inference")
            return {task_names[0]: list(kwargs.values()) if kwargs else list(args)}

        if all([isinstance(task_distr, int) for task_distr in distribution.values()]):
            ints = True
        elif all([isinstance(task_distr, list) for task_distr in distribution.values()]):
            ints = False
        else:
            raise ConfigError(
                f"Values of `{what_to_distribute}_distribution` attribute of `MultiTaskBert` have to be "
                f"either `int` or `list` not both. "
                f"{what_to_distribute}_distribution = {distribution}")

        args_by_task = {}
        
        flattened = []
        for task_name in task_names:
            if isinstance(task_name, str):
                flattened.append(task_name)
            else:
                flattened.extend(task_name)
        task_names = flattened 

        if args and not ints:
            ints = True
            distribution = {task_name: len(in_distr) for task_name, in_distr in distribution.items()}
        if ints:
            if kwargs:
                values = list(kwargs.values())
            else:
                values = args
            n_distributed = sum([n_args for n_args in distribution.values()])
            if len(values) != n_distributed:
                raise ConfigError(
                    f"The number of '{what_to_distribute}' arguments of MultitaskBert does not match "
                    f"the number of distributed params according to '{what_to_distribute}_distribution' parameter. "
                    f"{len(values)} parameters are in '{what_to_distribute}' and {n_distributed} parameters are "
                    f"required '{what_to_distribute}_distribution'. "
                    f"{what_to_distribute}_distribution = {distribution}")
            values_taken = 0
            for task_name in task_names:
                args_by_task[task_name] = {}
                n_args = distribution[task_name]
                args_by_task[task_name] = [values[i] for i in range(values_taken, values_taken + n_args)]
                values_taken += n_args
            
        else:
            assert kwargs
            arg_names_used = []
            for task_name in task_names:
                in_distr = distribution[task_name]
                args_by_task[task_name] = {}    
                args_by_task[task_name] = [kwargs[arg_name] for arg_name in in_distr]
                arg_names_used += in_distr
            set_used = set(arg_names_used)
            set_all = set(kwargs.keys())
            if set_used != set_all:
                raise ConfigError(f"There are unused '{what_to_distribute}' parameters {set_all - set_used}")
        return args_by_task

    def __call__(self, *args, **kwargs):
        """Calls one or several BERT heads depending on provided task names. ``args`` and ``kwargs`` contain 
        inputs of BERT tasks. ``args`` and ``kwargs`` cannot be used together. If ``args`` are used ``args`` content
        has to be

        .. code-block:: python
        
            args = [
                task1_in_x[0],
                task1_in_x[1],
                ...
                task2_in_x[0],
                task2_in_x[1],
                ...
            ]

        If ``kwargs`` are used and ``in_distribution`` is a dictionary of ``int``, then ``kwargs``' order has to be
        the same as ``args`` order described in the previous paragraph. If ``in_distribution`` is a dictionary of
        lists of ``str``, then all task names from ``in_distribution`` have to be present in ``kwargs`` keys.
        
        Returns:
            list of results of called tasks.
        """
        if self.inference_task_names is None:
            task_names = list(self.tasks.keys())
        else:
            task_names = self.inference_task_names
        if not task_names:
            raise ValueError("No tasks to call")
        if args and kwargs:
            raise ValueError("You may use either args or kwargs not both")
        return self.call(args, kwargs, task_names)

    def call(
            self, 
            args: Tuple[Any],
            kwargs: Dict[str, Any], 
            task_names: Optional[Union[List[str], str]],
            in_distribution: Optional[Union[Dict[str, int], Dict[str, List[str]]]] = None,
    ):
        """Calls one or several BERT heads depending on provided task names in ``task_names`` parameter. ``args`` and
        ``kwargs`` contain inputs of BERT tasks. ``args`` and ``kwargs cannot be used simultaneously. If ``args`` are
        used ``args``, content has to be
        
        .. code-block:: python

            args = [
                task1_in_x[0],
                task1_in_x[1],
                ...
                task2_in_x[0],
                task2_in_x[1],
                ...
            ]

        If ``kwargs`` is used ``kwargs`` keys has to match content of ``in_names`` params of called tasks.

        Args:
            args: generally, ``args`` parameter of ``__call__`` method of this component or ``MTBertReUser``. Inputs 
                of one or several tasks. Has to be empty if ``kwargs`` argument is used.
            kwargs: generally, ``kwargs`` parameter of ``__call__`` method of this component or ``MTBertReUser``. 
                Inputs of one or several tasks. Has to be empty if ``args`` argument is used.
            task_names: names of tasks that are called. If ``str``, then 1 task is called. If a task name is an
                element of ``task_names`` list, then this task is run independently. If task an element of
                ``task_names`` is an list of strings, then tasks in the inner list are run simultaneously.
            in_distribution: a distribution of variables from ``"in"`` config parameters between tasks. For details
                see method ``__init__`` docstring.
        
        Returns:
            list results of called tasks.
        """
        args_by_task = self._distribute_arguments_by_tasks(args, kwargs, task_names, "in", in_distribution)        
        results = []
        task_count = 0
        for elem in task_names:
            if isinstance(elem, str):
                task_count += 1
                task = self.tasks[elem]
                fetches, feed_dict = task.get_sess_run_infer_args(*args_by_task[elem])
                sess_run_res = self.sess.run(fetches, feed_dict=feed_dict)
                results.append(task.post_process_preds(sess_run_res))
            else:
                fetches = []
                for task_name in elem:
                    task_count += 1
                    feed_dict = {}
                    task_fetches, task_feed_dict = self.tasks[task_name].get_sess_run_infer_args(
                        *args_by_task[task_name])
                    fetches.append(task_fetches)
                    feed_dict = self._unite_task_feed_dicts(feed_dict, task_feed_dict, task_name)
                sess_run_res = self.sess.run(fetches, feed_dict=feed_dict)
                for task_name, srs in zip(elem, sess_run_res):
                    task_results = self.tasks[task_name].post_process_preds(srs)
                    results.append(task_results)
        if task_count == 1:
            results = results[0]
        return results


@register("mt_bert_reuser")
class MTBertReUser:
    """Instances of this class are for multi-task BERT inference. In inference config ``MultiTaskBert`` class may
    not perform inference of some tasks. For example, you may need to sequentially apply two models with BERT.
    In that case, ``mt_bert_reuser`` is created to call remaining tasks.

    Args:
        mt_bert: An instance of ``MultiTaskBert``
        task_names: Names of infered tasks. If ``task_names`` is ``str``, then ``task_names`` is the name of the only
            infered task. If ``task_names`` is ``list``, then its elements can be either strings or lists of strings.
            If an element of ``task_names`` is a string, then this element is a name of a task that is run 
            independently. If an element of ``task_names`` is a list of strings, then the element is a list of names
            of tasks that have common inputs and run simultaneously. For detailed information look up
            ``MultiTaskBert`` ``inference_task_names`` parameter.
    """
    def __init__(
            self, 
            mt_bert: MultiTaskBert, 
            task_names: Union[str, List[Union[str, List[str]]]], 
            in_distribution: Union[Dict[str, int], Dict[str, List[str]]] = None,
            *args, 
            **kwargs):
        self.mt_bert = mt_bert
        if isinstance(task_names, str):
            task_names = [task_names]
        elif not task_names:
            raise ValueError("At least 1 task has to specified")
        self.task_names = task_names
        flattened = []
        for elem in self.task_names:
            if isinstance(elem, str):
                flattened.append(elem)
            else:
                flattened.extend(elem)

        if in_distribution is None:
            if len(flattened) > 1:
                raise ValueError(
                    "If ``in_distribution`` parameter is not provided, there has to be only 1 task."
                    f"task_names = {self.task_names}")
          
        self.in_distribution = in_distribution

    def __call__(self, *args, **kwargs) -> List[Any]:
        """Infer tasks listed in parameter ``task_names``. One of parameters ``args`` and ``kwargs`` has to be empty.

        Args:
            args: inputs and labels of infered tasks.
            kwargs: inputs and labels of infered tasks.

        Returns:
            list of results of inference of tasks listed in ``task_names``
        """
        res = self.mt_bert.call(args, kwargs, task_names=self.task_names, in_distribution=self.in_distribution)
        return res


@register("input_splitter")
class InputSplitter:
    """The instance of these class in pipe splits a batch of sequences of identical length or dictionaries with 
    identical keys into tuple of batches.

    Args:
        keys_to_extract: a sequence of ints or strings that have to match keys of split dictionaries.
    """
    def __init__(self, keys_to_extract: Union[List[str], Tuple[str, ...]], **kwargs):
        self.keys_to_extract = keys_to_extract

    def __call__(self, inp: Union[List[dict], List[List[int]], List[Tuple[int]]]) -> List[list]:
        """Returns batches of values from ``inp``. Every batch contains values that have same key from 
        ``keys_to_extract`` attribute. The order of elements of ``keys_to_extract`` is preserved.

        Args:
            inp: A sequence of dictionaries with identical keys

        Returns:
            A list of lists of values of dictionaries from ``inp``
        """
        extracted = [[] for _ in self.keys_to_extract]
        for item in inp:
            for i, key in enumerate(self.keys_to_extract):
                extracted[i].append(item[key])
        return extracted

