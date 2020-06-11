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
from typing import List, Union, Dict, Optional

import numpy as np
import tensorflow as tf
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.layers.tf_layers import bi_rnn
from deeppavlov.core.models.tf_model import LRScheduledTFModel

log = getLogger(__name__)


def token_from_subtoken(units: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """ Assemble token level units from subtoken level units

    Args:
        units: tf.Tensor of shape [batch_size, SUBTOKEN_seq_length, n_features]
        mask: mask of token beginnings. For example: for tokens

                [[``[CLS]`` ``My``, ``capybara``, ``[SEP]``],
                [``[CLS]`` ``Your``, ``aar``, ``##dvark``, ``is``, ``awesome``, ``[SEP]``]]

            the mask will be

                [[0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0]]

    Returns:
        word_level_units: Units assembled from ones in the mask. For the
            example above this units will correspond to the following

                [[``My``, ``capybara``],
                [``Your`, ``aar``, ``is``, ``awesome``,]]

            the shape of this tensor will be [batch_size, TOKEN_seq_length, n_features]
    """
    shape = tf.cast(tf.shape(units), tf.int64)
    batch_size = shape[0]
    nf = shape[2]
    nf_int = units.get_shape().as_list()[-1]

    # number of TOKENS in each sentence
    token_seq_lengths = tf.cast(tf.reduce_sum(mask, 1), tf.int64)
    # for a matrix m =
    # [[1, 1, 1],
    #  [0, 1, 1],
    #  [1, 0, 0]]
    # it will be
    # [3, 2, 1]

    n_words = tf.reduce_sum(token_seq_lengths)
    # n_words -> 6

    max_token_seq_len = tf.cast(tf.reduce_max(token_seq_lengths), tf.int64)
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

    sample_ids_in_batch = tf.pad(idxs[:, 0], [[1, 0]])
    # for indices
    # [[0, 0],
    #  [0, 1]
    #  [0, 2],
    #  [1, 1],
    #  [1, 2],
    #  [2, 0]]
    # it is
    # [0, 0, 0, 0, 1, 1, 2]
    # padding is for computing change from one sample to another in the batch

    a = tf.cast(tf.not_equal(sample_ids_in_batch[1:], sample_ids_in_batch[:-1]), tf.int64)
    # for the example above the result of this statement equals
    # [0, 0, 0, 1, 0, 1]
    # so data samples begin in 3rd and 5th positions (the indexes of ones)

    # transforming sample start masks to the sample starts themselves
    q = a * tf.cast(tf.range(n_words), tf.int64)
    # [0, 0, 0, 3, 0, 5]
    count_to_substract = tf.pad(tf.boolean_mask(q, q), [(1, 0)])
    # [0, 3, 5]

    new_word_indices = tf.cast(tf.range(n_words), tf.int64) - tf.gather(count_to_substract, tf.cumsum(a))
    # tf.range(n_words) -> [0, 1, 2, 3, 4, 5]
    # tf.cumsum(a) -> [0, 0, 0, 1, 1, 2]
    # tf.gather(count_to_substract, tf.cumsum(a)) -> [0, 0, 0, 3, 3, 5]
    # new_word_indices -> [0, 1, 2, 3, 4, 5] - [0, 0, 0, 3, 3, 5] = [0, 1, 2, 0, 1, 0]
    # new_word_indices is the concatenation of range(word_len(sentence))
    # for all sentences in units

    n_total_word_elements = tf.cast(batch_size * max_token_seq_len, tf.int32)
    word_indices_flat = tf.cast(idxs[:, 0] * max_token_seq_len + new_word_indices, tf.int32)
    x_mask = tf.reduce_sum(tf.one_hot(word_indices_flat, n_total_word_elements), 0)
    x_mask = tf.cast(x_mask, tf.bool)
    # to get absolute indices we add max_token_seq_len:
    # idxs[:, 0] * max_token_seq_len -> [0, 0, 0, 1, 1, 2] * 2 = [0, 0, 0, 3, 3, 6]
    # word_indices_flat -> [0, 0, 0, 3, 3, 6] + [0, 1, 2, 0, 1, 0] = [0, 1, 2, 3, 4, 6]
    # total number of words in the batch (including paddings)
    # batch_size * max_token_seq_len -> 3 * 3 = 9
    # tf.one_hot(...) ->
    # [[1. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 1. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
    #  [0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    #  x_mask -> [1, 1, 1, 1, 1, 0, 1, 0, 0]

    full_range = tf.cast(tf.range(batch_size * max_token_seq_len), tf.int32)
    # full_range -> [0, 1, 2, 3, 4, 5, 6, 7, 8]
    nonword_indices_flat = tf.boolean_mask(full_range, tf.math.logical_not(x_mask))
    # # y_idxs -> [5, 7, 8]

    # get a sequence of units corresponding to the start subtokens of the words
    # size: [n_words, n_features]
    elements = tf.gather_nd(units, idxs)

    # prepare zeros for paddings
    # size: [batch_size * TOKEN_seq_length - n_words, n_features]
    paddings = tf.zeros(tf.stack([tf.reduce_sum(max_token_seq_len - token_seq_lengths),
                                  nf], 0), tf.float32)

    tensor_flat = tf.dynamic_stitch([word_indices_flat, nonword_indices_flat],
                                    [elements, paddings])
    # tensor_flat -> [x, x, x, x, x, 0, x, 0, 0]

    tensor = tf.reshape(tensor_flat, tf.stack([batch_size, max_token_seq_len, nf_int], 0))
    # tensor -> [[x, x, x],
    #            [x, x, 0],
    #            [x, 0, 0]]

    return tensor


@register('bert_sequence_network')
class BertSequenceNetwork(LRScheduledTFModel):
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
        ema_decay: what exponential moving averaging to use for network parameters, value from 0.0 to 1.0.
            Values closer to 1.0 put weight on the parameters history and values closer to 0.0 corresponds put weight
            on the current parameters.
        ema_variables_on_cpu: whether to put EMA variables to CPU. It may save a lot of GPU memory
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

    def __init__(self,
                 keep_prob: float,
                 bert_config_file: str,
                 pretrained_bert: str = None,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 encoder_layer_ids: List[int] = (-1,),
                 encoder_dropout: float = 0.0,
                 optimizer: str = None,
                 weight_decay_rate: float = 1e-6,
                 ema_decay: float = None,
                 ema_variables_on_cpu: bool = True,
                 freeze_embeddings: bool = False,
                 learning_rate: float = 1e-3,
                 bert_learning_rate: float = 2e-5,
                 min_learning_rate: float = 1e-07,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: float = 1.0,
                 **kwargs) -> None:
        super().__init__(learning_rate=learning_rate,
                         learning_rate_drop_div=learning_rate_drop_div,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         load_before_drop=load_before_drop,
                         clip_norm=clip_norm,
                         **kwargs)
        self.keep_prob = keep_prob
        self.encoder_layer_ids = encoder_layer_ids
        self.encoder_dropout = encoder_dropout
        self.optimizer = optimizer
        self.weight_decay_rate = weight_decay_rate
        self.ema_decay = ema_decay
        self.ema_variables_on_cpu = ema_variables_on_cpu
        self.freeze_embeddings = freeze_embeddings
        self.bert_learning_rate_multiplier = bert_learning_rate / learning_rate
        self.min_learning_rate = min_learning_rate

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

    def _init_graph(self) -> None:
        self.seq_lengths = tf.reduce_sum(self.y_masks_ph, axis=1)

        self.bert = BertModel(config=self.bert_config,
                              is_training=self.is_train_ph,
                              input_ids=self.input_ids_ph,
                              input_mask=self.input_masks_ph,
                              token_type_ids=self.token_types_ph,
                              use_one_hot_embeddings=False)

        with tf.variable_scope('ner'):
            layer_weights = tf.get_variable('layer_weights_',
                                            shape=len(self.encoder_layer_ids),
                                            initializer=tf.ones_initializer(),
                                            trainable=True)
            layer_mask = tf.ones_like(layer_weights)
            layer_mask = tf.nn.dropout(layer_mask, self.encoder_keep_prob_ph)
            layer_weights *= layer_mask
            # to prevent zero division
            mask_sum = tf.maximum(tf.reduce_sum(layer_mask), 1.0)
            layer_weights = tf.unstack(layer_weights / mask_sum)
            # TODO: may be stack and reduce_sum is faster
            units = sum(w * l for w, l in zip(layer_weights, self.encoder_layers()))
            units = tf.nn.dropout(units, keep_prob=self.keep_prob_ph)
        return units

    def _get_tag_mask(self) -> tf.Tensor:
        """
        Returns: tag_mask,
            a mask that selects positions corresponding to word tokens (not padding and `CLS`)
        """
        max_length = tf.reduce_max(self.seq_lengths)
        one_hot_max_len = tf.one_hot(self.seq_lengths - 1, max_length)
        tag_mask = tf.cumsum(one_hot_max_len[:, ::-1], axis=1)[:, ::-1]
        return tag_mask

    def encoder_layers(self):
        """
        Returns: the output of BERT layers specfied in ``self.encoder_layers_ids``
        """
        return [self.bert.all_encoder_layers[i] for i in self.encoder_layer_ids]

    def _init_placeholders(self) -> None:
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
        self.learning_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='learning_rate_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.encoder_keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='encoder_keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _init_optimizer(self) -> None:
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

    def get_train_op(self, loss: tf.Tensor, learning_rate: Union[tf.Tensor, float], **kwargs) -> tf.Operation:
        assert "learnable_scopes" not in kwargs, "learnable scopes unsupported"
        # train_op for bert variables
        kwargs['learnable_scopes'] = ('bert/encoder', 'bert/embeddings')
        if self.freeze_embeddings:
            kwargs['learnable_scopes'] = ('bert/encoder',)
        bert_learning_rate = learning_rate * self.bert_learning_rate_multiplier
        bert_train_op = super().get_train_op(loss,
                                             bert_learning_rate,
                                             **kwargs)
        # train_op for ner head variables
        kwargs['learnable_scopes'] = ('ner',)
        head_train_op = super().get_train_op(loss,
                                             learning_rate,
                                             **kwargs)
        return tf.group(bert_train_op, head_train_op)

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

    def _build_feed_dict(self, input_ids, input_masks, token_types=None, *args,  **kwargs):
        raise NotImplementedError("You must implement _build_feed_dict in your derived class.")

    def train_on_batch(self,
                       input_ids: Union[List[List[int]], np.ndarray],
                       input_masks: Union[List[List[int]], np.ndarray],
                       *args, **kwargs) -> Dict[str, float]:
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
        feed_dict = self._build_feed_dict(input_ids, input_masks, *args, **kwargs)

        if self.ema:
            self.sess.run(self.ema.switch_to_train_op)
        _, loss, lr = self.sess.run([self.train_op, self.loss, self.learning_rate_ph],
                                     feed_dict=feed_dict)
        return {'loss': loss,
                'head_learning_rate': float(lr),
                'bert_learning_rate': float(lr) * self.bert_learning_rate_multiplier}

    def __call__(self,
                 input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray],
                 **kwargs) -> Union[List[List[int]], List[np.ndarray]]:
        raise NotImplementedError("You must implement method __call__ in your derived class.")

    def save(self, exclude_scopes=('Optimizer', 'EMA/BackupVariables')) -> None:
        if self.ema:
            self.sess.run(self.ema.switch_to_train_op)
        return super().save(exclude_scopes=exclude_scopes)

    def load(self,
             exclude_scopes=('Optimizer',
                             'learning_rate',
                             'momentum',
                             'EMA/BackupVariables'),
             **kwargs) -> None:
        return super().load(exclude_scopes=exclude_scopes, **kwargs)


@register('bert_sequence_tagger')
class BertSequenceTagger(BertSequenceNetwork):
    """BERT-based model for text tagging. It predicts a label for every token (not subtoken) in the text.
    You can use it for sequence labeling tasks, such as morphological tagging or named entity recognition.
    See :class:`deeppavlov.models.bert.bert_sequence_tagger.BertSequenceNetwork`
    for the description of inherited parameters.

    Args:
        n_tags: number of distinct tags
        use_crf: whether to use CRF on top or not
        use_birnn: whether to use bidirection rnn after BERT layers.
            For NER and morphological tagging we usually set it to `False` as otherwise the model overfits
        birnn_cell_type: the type of Bidirectional RNN. Either `lstm` or `gru`
        birnn_hidden_size: number of hidden units in the BiRNN layer in each direction
        return_probas: set this to `True` if you need the probabilities instead of raw answers
    """

    def __init__(self,
                 n_tags: List[str],
                 keep_prob: float,
                 bert_config_file: str,
                 pretrained_bert: str = None,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 use_crf=False,
                 encoder_layer_ids: List[int] = (-1,),
                 encoder_dropout: float = 0.0,
                 optimizer: str = None,
                 weight_decay_rate: float = 1e-6,
                 use_birnn: bool = False,
                 birnn_cell_type: str = 'lstm',
                 birnn_hidden_size: int = 128,
                 ema_decay: float = None,
                 ema_variables_on_cpu: bool = True,
                 return_probas: bool = False,
                 freeze_embeddings: bool = False,
                 learning_rate: float = 1e-3,
                 bert_learning_rate: float = 2e-5,
                 min_learning_rate: float = 1e-07,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: float = 1.0,
                 **kwargs) -> None:
        self.n_tags = n_tags
        self.use_crf = use_crf
        self.use_birnn = use_birnn
        self.birnn_cell_type = birnn_cell_type
        self.birnn_hidden_size = birnn_hidden_size
        self.return_probas = return_probas
        super().__init__(keep_prob=keep_prob,
                         bert_config_file=bert_config_file,
                         pretrained_bert=pretrained_bert,
                         attention_probs_keep_prob=attention_probs_keep_prob,
                         hidden_keep_prob=hidden_keep_prob,
                         encoder_layer_ids=encoder_layer_ids,
                         encoder_dropout=encoder_dropout,
                         optimizer=optimizer,
                         weight_decay_rate=weight_decay_rate,
                         ema_decay=ema_decay,
                         ema_variables_on_cpu=ema_variables_on_cpu,
                         freeze_embeddings=freeze_embeddings,
                         learning_rate=learning_rate,
                         bert_learning_rate=bert_learning_rate,
                         min_learning_rate=min_learning_rate,
                         learning_rate_drop_div=learning_rate_drop_div,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         load_before_drop=load_before_drop,
                         clip_norm=clip_norm,
                         **kwargs)

    def _init_graph(self) -> None:
        self._init_placeholders()

        units = super()._init_graph()

        with tf.variable_scope('ner'):
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

    def _init_placeholders(self) -> None:
        super()._init_placeholders()
        self.y_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_ph')
        self.y_masks_ph = tf.placeholder(shape=(None, None),
                                         dtype=tf.int32,
                                         name='y_mask_ph')

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

    def _build_feed_dict(self, input_ids, input_masks, y_masks, y=None):
        feed_dict = self._build_basic_feed_dict(input_ids, input_masks, train=(y is not None))
        feed_dict[self.y_masks_ph] = y_masks
        if y is not None:
            feed_dict[self.y_ph] = y
        return feed_dict

    def __call__(self,
                 input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray],
                 y_masks: Union[List[List[int]], np.ndarray]) -> Union[List[List[int]], List[np.ndarray]]:
        """ Predicts tag indices for a given subword tokens batch

        Args:
            input_ids: indices of the subwords
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word

        Returns:
            Label indices or class probabilities for each token (not subtoken)

        """
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
