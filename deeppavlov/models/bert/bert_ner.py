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
from logging import getLogger
from typing import List, Any, Tuple

import tensorflow as tf
from tensorflow.python.ops import array_ops
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.tf_model import LRScheduledTFModel

log = getLogger(__name__)


@register('bert_ner')
class BertNerModel(LRScheduledTFModel):
    """Bert-based model for text named entity tagging.

    Uses bert token representation to predict it's bio tag.
    Representation is obtained by averaging several hidden layers from bert encoder.
    Ner head consists of linear layers.
￼
￼   Args:
￼       bert_config_file: path to Bert configuration file
￼       n_tags: number of distinct tags
￼       keep_prob: dropout keep_prob for non-Bert layers
￼       attention_probs_keep_prob: keep_prob for Bert self-attention layers
￼       hidden_keep_prob: keep_prob for Bert hidden layers
        encoder_layer_ids: list of averaged layers from Bert encoder (layer ids)
￼       return_probas: set True if return class probabilites instead of most probable label needed
￼       optimizer: name of tf.train.* optimizer or None for `AdamWeightDecayOptimizer`
￼       num_warmup_steps:
￼       weight_decay_rate: L2 weight decay for `AdamWeightDecayOptimizer`
￼       pretrained_bert: pretrained Bert checkpoint
￼       min_learning_rate: min value of learning rate if learning rate decay is used
￼   """

    # TODO: add warmup
    # TODO: add head-only pre-training
    def __init__(self,
                 bert_config_file: str,
                 n_tags: List[str],
                 keep_prob: float,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 encoder_layer_ids: List[int] = tuple(range(12)),
                 optimizer: str = None,
                 num_warmup_steps: int = None,
                 focal_alpha: float = None,
                 focal_gamma: float = None,
                 ema_decay: float = None,
                 ema_variables_on_cpu: bool = True,
                 weight_decay_rate: float = 0.01,
                 return_probas: bool = False,
                 pretrained_bert: str = None,
                 head_learning_rate_mult = 1.0,
                 min_learning_rate: float = 1e-06,
                 use_crf=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.return_probas = return_probas
        self.n_tags = n_tags
        self.head_learning_rate_mult = head_learning_rate_mult
        self.min_learning_rate = min_learning_rate
        self.keep_prob = keep_prob
        self.optimizer = optimizer
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.ema_decay = ema_decay
        self.ema_variables_on_cpu = ema_variables_on_cpu
        self.encoder_layer_ids = encoder_layer_ids
        self.num_warmup_steps = num_warmup_steps
        self.weight_decay_rate = weight_decay_rate
        self.use_crf = use_crf

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
                and not tf.train.checkpoint_exists(str(self.load_path.resolve())):
            log.info('[initializing model with Bert from {}]'.format(pretrained_bert))
            # Exclude optimizer and classification variables from saved variables
            var_list = self._get_saveable_variables(
                exclude_scopes=('Optimizer', 'learning_rate', 'momentum', 'ner', 'EMA'))
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, pretrained_bert)

        if self.load_path is not None:
            self.load()

        if self.ema:
            self.sess.run(self.ema.init_op)

    def _init_graph(self):
        self._init_placeholders()

        self.bert = BertModel(config=self.bert_config,
                              is_training=self.is_train_ph,
                              input_ids=self.input_ids_ph,
                              input_mask=self.input_masks_ph,
                              token_type_ids=self.token_types_ph,
                              use_one_hot_embeddings=False)

        encoder_layers = [self.bert.all_encoder_layers[i]
                          for i in self.encoder_layer_ids]

        with tf.variable_scope('ner'):
            layer_weights = tf.get_variable('layer_weights_',
                                            shape=len(encoder_layers),
                                            initializer=tf.ones_initializer(),
                                            trainable=True)
            layer_weights = tf.unstack(layer_weights / len(encoder_layers))
            # TODO: may be stack and reduce_sum is faster
            output_layer = sum(w * l for w, l in zip(layer_weights, encoder_layers))
            output_layer = tf.nn.dropout(output_layer, keep_prob=self.keep_prob_ph)
            # TODO: maybe add one more layer?
            logits = tf.layers.dense(output_layer, units=self.n_tags, name="output_dense")

            self.logits = self.token_from_subtoken(logits, self.y_masks_ph)

            self.seq_lengths = tf.reduce_sum(self.y_masks_ph, axis=1)
            max_length = tf.reduce_max(self.seq_lengths)
            one_hot_max_len = tf.one_hot(self.seq_lengths - 1, max_length)
            tag_mask = tf.cumsum(one_hot_max_len[:, ::-1], axis=1)[:, ::-1]

            # CRF
            if self.use_crf:
                transition_params = tf.get_variable('Transition_Params',
                                                    shape=[self.n_tags,
                                                           self.n_tags],
                                                    initializer=tf.zeros_initializer())
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.logits,
                                                                                      self.y_ph,
                                                                                      self.seq_lengths,
                                                                                      transition_params)
                loss_tensor = -log_likelihood
                self._transition_params = transition_params

            self.y_predictions = tf.argmax(self.logits, -1)
            self.y_probas = tf.nn.softmax(self.logits, axis=2)

        with tf.variable_scope("loss"):
            y_mask = tf.cast(tag_mask, tf.float32)
            if self.use_crf:
                self.loss = tf.reduce_mean(loss_tensor)
            elif (self.focal_alpha is None) or (self.focal_gamma is None):
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y_ph,
                                                                   logits=self.logits,
                                                                   weights=y_mask)
            else:
                y_onehot = tf.one_hot(self.y_ph, self.n_tags)
                self.loss = self.focal_loss(labels=y_onehot,
                                            probs=self.y_probas,
                                            weights=y_mask,
                                            alpha=self.focal_alpha,
                                            gamma=self.focal_gamma)

    @staticmethod
    def focal_loss(labels, probs, weights=None, alpha=1.0, gamma=1):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         labels: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         probs: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         weights: A float tensor of shape [batch_size, num_anchors]
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """
        labels = tf.cast(labels, tf.float32)
        probs = tf.cast(probs, tf.float32)

        zeros = array_ops.zeros_like(probs, dtype=probs.dtype)

        # For positive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = array_ops.where(labels > zeros, labels - probs, zeros)

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = array_ops.where(labels > zeros, zeros, probs)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * \
            tf.log(tf.clip_by_value(probs, 1e-8, 1.0)) \
            - (1 - alpha) * (neg_p_sub ** gamma) * \
            tf.log(tf.clip_by_value(1.0 - probs, 1e-8, 1.0))
        if weights is not None:
            per_entry_cross_ent = tf.multiply(per_entry_cross_ent,
                                              tf.expand_dims(weights, -1))
        return tf.reduce_sum(per_entry_cross_ent)

    def _init_placeholders(self):
        self.input_ids_ph = tf.placeholder(shape=(None, None),
                                           dtype=tf.int32,
                                           name='token_indices_ph')
        self.input_masks_ph = tf.placeholder(shape=(None, None),
                                             dtype=tf.int32,
                                             name='token_mask_ph')
        self.token_types_ph = \
                tf.placeholder_with_default(tf.zeros_like(self.input_ids_ph, dtype=tf.int32),
                                            shape=self.input_ids_ph.shape,
                                            name='token_types_ph')

        self.y_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_ph')
        self.y_masks_ph = tf.placeholder(shape=(None, None),
                                         dtype=tf.int32,
                                         name='y_mask_ph')

        self.learning_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='learning_rate_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _init_optimizer(self):
        with tf.variable_scope('Optimizer'):
            self.global_step = tf.get_variable('global_step',
                                               shape=[],
                                               dtype=tf.int32,
                                               initializer=tf.constant_initializer(0),
                                               trainable=False)
            # default optimizer for Bert is Adam with fixed L2 regularization

        if self.optimizer is None:
            self.train_op = \
                self.get_train_op(self.loss,
                                  learning_rate=self.learning_rate_ph,
                                  optimizer=AdamWeightDecayOptimizer,
                                  weight_decay_rate=self.weight_decay_rate,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=1e-6,
                                  optimizer_scope_name='Optimizer',
                                  exclude_from_weight_decay=["LayerNorm",
                                                             "layer_norm",
                                                             "bias",
                                                             "EMA"])
        else:
            self.train_op = self.get_train_op(self.loss,
                                              learning_rate=self.learning_rate_ph,
                                              optimizer_scope_name='Optimizer')

        if self.optimizer is None:
            with tf.variable_scope('Optimizer'):
                new_global_step = self.global_step + 1
                self.train_op = tf.group(self.train_op, [self.global_step.assign(new_global_step)])

        if self.ema_decay is not None:
            _vars = self._get_trainable_variables(exclude_scopes=["Optimizer",
                                                                  "LayerNorm",
                                                                  "layer_norm",
                                                                  "bias",
                                                                  "learning_rate",
                                                                  "momentum"])

            self.ema = ExponentialMovingAverage(self.ema_decay,
                                                variables_on_cpu=self.ema_variables_on_cpu)
            self.train_op = self.ema.build(self.train_op, _vars, name="EMA")
        else:
            self.ema = None

    def get_train_op(self, loss, learning_rate, **kwargs):
        assert "learnable_scopes" not in kwargs, "learnable scopes unsupported"
        # train_op for bert variables
        kwargs['learnable_scopes'] = ('(?!ner)',)
        bert_train_op = super().get_train_op(loss,
                                             learning_rate,
                                             **kwargs)
        # train_op for ner head variables
        kwargs['learnable_scopes'] = ('ner',)
        head_learning_rate = learning_rate * self.head_learning_rate_mult
        head_train_op = super().get_train_op(loss,
                                             head_learning_rate,
                                             **kwargs)
        return tf.group(bert_train_op, head_train_op)

    @staticmethod
    def token_from_subtoken(units, mask):
        """ Assemble token level units from subtoken level units

        Args:
            units: tf.Tensor of shape [batch_size, SUBTOKEN_seq_length, n_features]
            mask: mask of startings of new tokens. Example: for tokens
                [['[CLS]' 'My', 'capybara', '[SEP]'],
                 ['[CLS]' 'Your', 'aar', '##dvark', 'is', 'awesome', '[SEP]']]
                the mask will be
                [[0, 1, 1, 0, 0, 0, 0],
                 [0, 1, 1, 0, 1, 1, 0]]

        Returns:
            word_level_units: Units assembled from ones in the mask. For the
                example above this units will correspond to the following

                [['My', 'capybara'],
                 ['Your', 'aar', 'is', 'awesome',]]

                the shape of this thesor will be [batch_size, TOKEN_seq_length, n_features]
        """
        shape = tf.cast(tf.shape(units), tf.int64)
        bs = shape[0]
        nf = shape[2]
        nf_int = units.get_shape().as_list()[-1]

        # numer of TOKENS in each sentence
        token_seq_lenghs = tf.cast(tf.reduce_sum(mask, 1), tf.int64)
        # for a matrix m =
        # [[1, 1, 1],
        #  [0, 1, 1],
        #  [1, 0, 0]]
        # it will be
        # [3, 2, 1]

        n_words = tf.reduce_sum(token_seq_lenghs)
        # n_words -> 6

        max_token_seq_len = tf.reduce_max(token_seq_lenghs)
        max_token_seq_len = tf.cast(max_token_seq_len, tf.int64)
        # max_token_seq_len -> 3

        idxs = tf.where(mask)
        # for the matrix mentioned above
        # tf.where(mask) ->
        # [[0, 0],
        #  [0, 1]
        #  [0, 2],
        #  [1, 1],
        #  [1, 2]
        #  [2, 0]]

        sample_id_in_batch = tf.pad(idxs[:, 0], [[1, 0]])
        # for indices
        # [[0, 0],
        #  [0, 1]
        #  [0, 2],
        #  [1, 1],
        #  [1, 2],
        #  [2, 0]]
        # it will be
        # [0, 0, 0, 0, 1, 1, 2]
        # padding is for computing change from one sample to another in the batch

        a = tf.cast(tf.not_equal(sample_id_in_batch[1:], sample_id_in_batch[:-1]), tf.int64)
        # for the example above the result of this line will be
        # [0, 0, 0, 1, 0, 1]
        # so the number of the sample in batch changes only in the last word element

        q = a * tf.cast(tf.range(n_words), tf.int64)
        # [0, 0, 0, 3, 0, 5]

        count_to_substract = tf.pad(tf.boolean_mask(q, q), [(1, 0)])
        # [0, 3, 5]

        new_word_indices = tf.cast(tf.range(n_words), tf.int64) - tf.gather(count_to_substract, tf.cumsum(a))
        # tf.range(n_words) -> [0, 1, 2, 3, 4, 5]
        # tf.cumsum(a) -> [0, 0, 0, 1, 1, 2]
        # tf.gather(count_to_substract, tf.cumsum(a)) -> [0, 0, 0, 3, 3, 5]
        # new_word_indices -> [0, 1, 2, 3, 4, 5] - [0, 0, 0, 3, 3, 5] = [0, 1, 2, 0, 1, 0]
        # this is new indices token dimension

        n_total_word_elements = tf.cast(bs * max_token_seq_len, tf.int32)
        x_mask = tf.reduce_sum(tf.one_hot(idxs[:, 0] * max_token_seq_len + new_word_indices, n_total_word_elements), 0)
        x_mask = tf.cast(x_mask, tf.bool)
        # to get absolute indices we add max_token_seq_len:
        # idxs[:, 0] * max_token_seq_len -> [0, 0, 0, 1, 1, 2] * 2 = [0, 0, 0, 3, 3, 6]
        # idxs[:, 0] * max_token_seq_len + new_word_indices ->
        # [0, 0, 0, 3, 3, 6] + [0, 1, 2, 0, 1, 0] = [0, 1, 2, 3, 4, 6]
        # total number of words in the batch (including paddings)
        # bs * max_token_seq_len -> 3 * 2 = 6
        # tf.one_hot(...) ->
        # [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
        #  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
        #  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
        #  [0. 0. 0. 1. 0. 0. 0. 0. 0.]
        #  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
        #  [0. 0. 0. 0. 0. 0. 1. 0. 0.]]
        #  x_mask -> [1, 1, 1, 1, 1, 0, 1, 0, 0]

        # full_range -> [0, 1, 2, 3, 4, 5, 6, 7, 8]
        full_range = tf.cast(tf.range(bs * max_token_seq_len), tf.int32)

        x_idxs = tf.boolean_mask(full_range, x_mask)
        # x_idxs -> [0, 1, 2, 3, 4, 6]

        y_mask = tf.math.logical_not(x_mask)
        y_idxs = tf.boolean_mask(full_range, y_mask)
        # y_idxs -> [5, 7, 8]

        # get a sequence of units corresponding to the start subtokens of the words
        # size: [n_words, n_features]
        els = tf.gather_nd(units, idxs)

        # prepare zeros for paddings
        # size: [batch_size * TOKEN_seq_length - n_words, n_features]
        paddings = tf.zeros(tf.stack([tf.reduce_sum(max_token_seq_len - token_seq_lenghs),
                                      nf], 0), tf.float32)

        tensor_flat = tf.dynamic_stitch([x_idxs, y_idxs], [els, paddings])
        # tensor_flat -> [x, x, x, x, x, 0, x, 0, 0]

        tensor = tf.reshape(tensor_flat, tf.stack([bs, max_token_seq_len, nf_int], 0))
        # tensor_flat -> [[x, x, x],
        #                 [x, x, 0],
        #                 [x, 0, 0]]

        return tensor

    def _build_feed_dict(self, input_ids, input_masks, y_masks, token_types=None, y=None):
        feed_dict = {
            self.input_ids_ph: input_ids,
            self.input_masks_ph: input_masks,
            self.y_masks_ph: y_masks
        }
        if token_types is not None:
            feed_dict[self.token_types_ph] = token_types
        if y is not None:
            feed_dict.update({
                self.y_ph: y,
                self.learning_rate_ph: max(self.get_learning_rate(), self.min_learning_rate),
                self.keep_prob_ph: self.keep_prob,
                self.is_train_ph: True,
            })

        return feed_dict

    def train_on_batch(self,
                       input_ids: List[List[int]],
                       input_masks: List[List[int]],
                       y_masks: List[List[int]],
                       y: List[List[int]]) -> dict:
        # for ids, masks, ys in zip(input_ids, input_masks, y):
        #     assert len(ids) == len(masks) == len(ys), \
        #         f"ids({len(ids)}) = {ids}, masks({len(masks)}) = {masks},"\
        #         f" ys({len(ys)}) = {ys} should have the same length."

        feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks, y=y)

        if self.ema:
            self.sess.run(self.ema.switch_to_train_op)
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return {'loss': loss, 'learning_rate': feed_dict[self.learning_rate_ph]}

    def __call__(self,
                 input_ids: List[List[int]],
                 input_masks: List[List[int]],
                 y_masks: List[List[int]]):
        for ids, masks in zip(input_ids, input_masks):
            assert len(ids) == len(masks), \
                f"ids({len(ids)}) = {ids}, masks({len(masks)}) = {masks}" \
                f" should have the same length."

        feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks)
        if self.ema:
            self.sess.run(self.ema.switch_to_test_op)
        if not self.return_probas:
            if self.use_crf:
                pred = self._decode_crf(feed_dict)
            else:
                pred, seq_lengths = self.sess.run([self.y_predictions, self.seq_lengths], feed_dict=feed_dict)
                pred = [p[:l] for l, p in zip(seq_lengths, pred)]
        else:
            pred = self.sess.run(self.y_probas, feed_dict=feed_dict)
        return pred

    def _decode_crf(self, feed_dict):
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


class MaskCutter(Component):
    def __init__(self, **kwargs):
        pass

    def __call__(self,
                 samples: List[List[Any]],
                 masks: List[List[int]]):
        samples_cut = []
        for s_list, m_list in zip(samples, masks):
            samples_cut.append([])
            for j in range(len(s_list)):
                if m_list[j]:
                    samples_cut[-1].append(s_list[j])
        return samples_cut


class ExponentialMovingAverage:
    def __init__(self,
                 decay: float = 0.999,
                 variables_on_cpu: bool = True) -> None:
        self.decay = decay
        self.ema = tf.train.ExponentialMovingAverage(decay=decay)
        self.var_device_name = '/cpu:0' if variables_on_cpu else None
        self.train_mode = None

    def build(self,
              minimize_op: tf.Tensor,
              update_vars: List[tf.Variable] = None,
              name: str = "EMA") -> tf.Tensor:
        with tf.variable_scope(name):
            if update_vars is None:
                update_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            with tf.control_dependencies([minimize_op]):
                minimize_op = self.ema.apply(update_vars)

            with tf.device(self.var_device_name):
                # Make backup variables
                with tf.variable_scope('BackupVariables'):
                    backup_vars = [tf.get_variable(var.op.name,
                                                   dtype=var.value().dtype,
                                                   trainable=False,
                                                   initializer=var.initialized_value())
                                   for var in update_vars]

                def ema_to_weights():
                    return tf.group(*(tf.assign(var, self.ema.average(var).read_value())
                                      for var in update_vars))

                def save_weight_backups():
                    return tf.group(*(tf.assign(bck, var.read_value())
                                      for var, bck in zip(update_vars, backup_vars)))

                def restore_weight_backups():
                    return tf.group(*(tf.assign(var, bck.read_value())
                                      for var, bck in zip(update_vars, backup_vars)))

                train_switch_op = restore_weight_backups()
                with tf.control_dependencies([save_weight_backups()]):
                    test_switch_op = ema_to_weights()

            self.train_switch_op = train_switch_op
            self.test_switch_op = test_switch_op
            self.do_nothing_op = tf.no_op()

        return minimize_op

    @property
    def init_op(self) -> tf.Operation:
        self.train_mode = False
        return self.test_switch_op

    @property
    def switch_to_train_op(self) -> tf.Operation:
        assert self.train_mode is not None, "ema variables aren't initialized"
        if not self.train_mode:
            # log.info("switching to train mode")
            self.train_mode = True
            return self.train_switch_op
        return self.do_nothing_op

    @property
    def switch_to_test_op(self) -> tf.Operation:
        assert self.train_mode is not None, "ema variables aren't initialized"
        if self.train_mode:
            # log.info("switching to test mode")
            self.train_mode = False
            return self.test_switch_op
        return self.do_nothing_op
